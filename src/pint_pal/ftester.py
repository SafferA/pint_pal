""" This is a set of utilities which performs F-test calculations for parameter inclusion/exclusion.

The F-tests are run with `pint_pal.parallel_fit.TestFitter`, which fits a set of nested timing models: one for
each parameter (or linked group of parameters) to be added to or removed from the timing model, plus a
baseline fit with no parameters added or removed. Each test fit is compared with the baseline fit using
`pint.utils.FTest()`.

`runFtest_parallel()` runs the fits in parallel on several CPUs, and `run_Ftests()` runs exactly the same fits
one at a time, for computers that aren't set up for parallel processing. Both process the results in the
same way and return the same outputs.
"""

import concurrent.futures
import json
import os
import re
import warnings

import numpy as np
from pint.utils import FTest

import pint_pal.parallel_fit as par_fit

ALPHA_ADD = 0.0027  # 3-sigma; parameters are recommended to be added if the F-statistic is <= ALPHA_ADD
ALPHA_REMOVE = 0.1336  # 1.5-sigma; parameters are recommended to be removed if the F-statistic is > ALPHA_REMOVE
ALPHA = ALPHA_ADD  # default significance value


def param_check(name, fitter, check_enabled=True):
    """
    Checks if a timing model parameter is in the timing model.

    Inputs:
    --------
    name [string]: Name of the timing model parameter to be checked.
    fitter [object]: The PINT fitter object.
    check_enabled [boolean]: If True, check if the model parameter is fit for as well [default: True].

    Returns:
    --------
    [boolean]: Returns either True or False depending on if the parameters is in the model and/or enabled.
    """
    if name not in fitter.model.params:
        return False
    # If frozen is False, then it's fit for, if frozen is True, parameter not fit for
    if check_enabled and getattr(fitter.model, name).frozen:
        return False
    return True


def get_fblist(fitter):
    """
    Returns the list of FB parameters for the ELL1 model.

    Input:
    --------
    fitter [object]: The PINT fitter object.

    Returns:
    --------
    fblist [dictionary]: Dictionary of FB parameters currently in the timing model, keyed by FB index.
    """
    fblist = {}
    for p in fitter.model.params:
        # Skip FB jumps (FBJ*) and parameters that have no value set
        if p.startswith("FB") and not p.startswith("FBJ") and getattr(fitter.model, p).value is not None:
            fblist[int(p[2:]) if len(p) > 2 else 0] = p
    # FB parameters should be a contiguous series FB0, FB1, ..., FBn
    if sorted(fblist) != list(range(len(fblist))):
        warnings.warn("FB parameters not a series of integers: " +
                      " ".join(fblist[i] for i in sorted(fblist)))
    return fblist


def _valid_ft(ft):
    """
    Returns True if the F-statistic is a usable number (not None, False, or NaN). E.g. PINT's FTest returns
    NaN when the two models have the same degrees of freedom. An F-statistic of exactly 0 is valid: it means
    the difference is so significant that the probability underflows.
    """
    if ft is None or isinstance(ft, (bool, np.bool_)):
        return False
    return bool(np.isfinite(ft))


def _recommendation_statements(add_params, remove_params):
    """ Returns the human-readable statements of which parameters are recommended to be added/removed. """
    if add_params:
        add_statement = "F-tests recommend adding the following parameters: " + " ".join(add_params)
    else:
        add_statement = "F-tests do not recommend adding any parameters."
    if remove_params:
        remove_statement = "F-tests recommend removing the following parameters: " + " ".join(remove_params)
    else:
        remove_statement = "F-tests do not recommend removing any parameters."
    return add_statement, remove_statement


# ---------------------------------------------------------------------------------------------------------
# Running the F-test fits
# ---------------------------------------------------------------------------------------------------------

class _SerialExecutor(concurrent.futures.Executor):
    """
    Executor that runs each task immediately in the current process. Passing this to `TestFitter.run()` makes
    PINT run the F-test fits one at a time through exactly the same code as the parallel workers (a fresh copy
    of the fitter for each test, and the same handling of failed fits), so the serial and parallel F-tests
    give identical results.
    """

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)
        return future


def _run_test_fits(fitter, executor=None, ncpu=None, maxiter=20, FDnparams=5, FBnparams=5):
    """
    Runs the baseline fit and all of the F-test fits with `parallel_fit.TestFitter`.

    Input:
    ----------
    fitter [object]: The PINT fitter object. Its current model and TOAs are what is tested.
    executor [object]: `concurrent.futures.Executor` to run the fits with. If None, one is created with ncpu workers.
    ncpu [int]: Number of CPUs to run the fits on, if executor is None.
    maxiter [int]: Maximum number of fit iterations for each test [default: 20].
    FDnparams [int]: Maximum number of FD parameters to test [default: 5].
    FBnparams [int]: Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].

    Returns:
    ---------
    results [dict]: Output of the `TestFitter` run, including the 'baseline' fit.
    """
    # Pass the fitter's current model and TOAs (rather than the original .par/.tim files) to test exactly what
    # has been fit, including any TOA excision. TestFitter includes the baseline fit in its tests
    try:
        tf = par_fit.TestFitter(model=fitter.model, toas=fitter.toas, maxFB=FBnparams, maxFD=FDnparams)
    except TypeError as e:
        # Older versions of parallel_fit only take parfile/timfile (and lack the baseline fit and nested FD/FB tests)
        if "model" in str(e) or "toas" in str(e):
            raise ImportError(
                f"The pint_pal.parallel_fit being imported ({par_fit.__file__}) is an older version whose "
                "TestFitter doesn't accept model=/toas=. Check that the updated file is named parallel_fit.py, "
                "that pint_pal is installed from the folder you updated (e.g. pip install -e), and restart the "
                "Python kernel.") from e
        raise
    return tf.run(executor=executor, ncpu=ncpu, maxiter=maxiter)


# ---------------------------------------------------------------------------------------------------------
# Processing the F-test results
# ---------------------------------------------------------------------------------------------------------

def _chi2_problem(chi2_simple, chi2_complex):
    """
    Returns a description of the problem if the more complex of two nested fits has a noticeably higher chi2
    than the simpler one, which can't happen when both fits are converged; otherwise returns None. PINT's
    FTest() returns 1.0 in that case, which would wrongly recommend removing a parameter.
    """
    tolerance = max(1e-3, 1e-6 * abs(chi2_simple))
    if chi2_complex > chi2_simple + tolerance:
        return "the fit with more parameters has a higher chi2, so a fit (often the baseline) is not converged"
    return None


def _evaluate_tests(results):
    """
    Computes the F-statistic of each test against the baseline fit.

    Input:
    ----------
    results [dict]: Output of `run_Ftests()` or `runFtest_parallel()`.

    Returns:
    ---------
    baseline [dict]: The baseline fit, with keys 'chi2' and 'dof'.
    tests [list]: One dictionary per test, with keys 'action' ('add' or 'remove'), 'params' (list of parameter
        names), 'chi2', 'dof', 'ft' (the F-statistic, or None if it can't be used) and 'problem' (why it
        can't be used, or None).
    """
    if 'baseline' not in results or not np.isfinite(results['baseline']['chi2']):
        raise ValueError("F-test results have no valid 'baseline' fit to compare against.")
    baseline = results['baseline']

    tests = []
    for key, test in results.items():
        # Skip the run metadata and the baseline fit, everything else is a test
        if key in ('meta', 'baseline'):
            continue
        # The test string is of the form "Combination <n>: <add/remove> <PAR1> and <PAR2> ..."
        action, parList = test['string'].split(':', 1)[1].strip().split(" ", 1)
        chi2 = test['chi2']
        dof = test['dof']
        # The simpler and more complex of the two models being compared
        simple, complex_ = ((chi2, dof), (baseline['chi2'], baseline['dof'])) if action == 'remove' else \
            ((baseline['chi2'], baseline['dof']), (chi2, dof))
        # A failed fit returns a NaN chi2; FTest() would then return 1.0, which would wrongly recommend removal
        if not np.isfinite(chi2):
            problem = "the fit failed"
        else:
            problem = _chi2_problem(simple[0], complex_[0])
        # FTest(chi2_1, dof_1, chi2_2, dof_2) compares the simpler model (1) with the more complex model (2)
        ft = None if problem else FTest(simple[0], simple[1], complex_[0], complex_[1])
        tests.append({'action': action, 'params': parList.split(" and "), 'chi2': chi2, 'dof': dof, 'ft': ft,
                      'problem': problem})
    return baseline, tests


def _table_lines(baseline, tests, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE):
    """
    Returns the F-test results as lines of a human-readable table. Each test is marked significant ('***')
    against the threshold for that kind of test, and failed fits are marked 'xxx'. The last line explains
    the markers.
    """
    lines = ["%42s %9s %5s %s" % ("", "Chi2", "NDOF", "Ftest"),
             "%42s %9.2f %5d --" % ("baseline", baseline['chi2'], baseline['dof'])]
    for test in tests:
        label = f"{test['action']} {', '.join(test['params'])}"
        ft = test['ft']
        if not _valid_ft(ft):
            lines.append("%42s %9.2f %5d xxx" % (label, test['chi2'], test['dof']))
            continue
        alpha = alpha_remove if test['action'] == 'remove' else alpha_add
        lines.append("%42s %9.2f %5d %.3g%s" % (label, test['chi2'], test['dof'], ft, " ***" if ft < alpha else ""))
    # Explain the markers, since '***' means the opposite outcome for 'add' and 'remove' tests
    lines.append(f"*** = significant: worth adding (F < {alpha_add:g}) or worth keeping (F < {alpha_remove:g}); "
                 "xxx = fit failed or not converged, or F-test not possible")
    return lines


def get_Ftest_lines(Ftest_dict, fitter=None, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE, alpha=None):
    """
    Returns the F-test results as lines of a human-readable table (including the header line), e.g. to write
    to a summary pdf.

    Input:
    ----------
    Ftest_dict [dict]: Output of `run_Ftests()` or `runFtest_parallel()`.
    fitter [object]: The PINT fitter object. [currently unused]
    alpha_add [float]: Add tests are marked significant ('***') if the F-statistic is lower than alpha_add [default: 0.0027].
    alpha_remove [float]: Removal tests are marked significant ('***') if the F-statistic is lower than
        alpha_remove [default: 0.1336].
    alpha [float]: Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns:
    ---------
    lines [list]: Lines of the table, as strings.
    """
    if alpha is not None:
        alpha_add = alpha
    baseline, tests = _evaluate_tests(Ftest_dict)
    return _table_lines(baseline, tests, alpha_add=alpha_add, alpha_remove=alpha_remove)


def _step_through_nested(chain, action, baseline, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE):
    """
    Steps through a nested series of FD or FB tests one term at a time, as repeated runs of the F-tests would.

    Each step compares a test with the previous (smaller) test in the series, or with the baseline for the
    first step, so each term is tested on its own given the terms before it. For removals, e.g. with FD1 and
    FD2 in the model: FD2 is tested against the baseline ("remove FD2"), then FD1 is tested by comparing
    "remove FD1 and FD2" with "remove FD2" (the FD1-only model), which is the test a second run would make.
    Stepping stops at the first term that fails, so a term that is needed on its own is never added or
    removed just because it was tested together with others.

    Input:
    ----------
    chain [list]: The nested tests (as returned by `_evaluate_tests()`), sorted by number of parameters.
    action [string]: 'add' or 'remove'.
    baseline [dict]: The baseline fit, with keys 'chi2' and 'dof'.
    alpha_add [float]: F-statistic threshold for adding parameters [default: 0.0027].
    alpha_remove [float]: F-statistic threshold for removing parameters [default: 0.1336].

    Returns:
    ---------
    chosen [list]: Parameters to add or remove (empty if the first step fails or the tests aren't nested).
    steps [list]: (new parameters, F-statistic, passed) for each step taken.
    """
    chosen = []
    previous = baseline
    steps = []
    for test in chain:
        # The tests should be nested, each one containing all of the parameters of the previous one
        # (if they aren't, any recommendation could break the series, so none is made)
        if not set(chosen) <= set(test['params']):
            warnings.warn(f"The {action} tests for {', '.join(test['params'])} are not nested, so no change is "
                          "recommended for them. Check that parallel_fit.py uses the instance's linked parameters.")
            return [], steps
        new_pars = [p for p in test['params'] if p not in chosen]
        # A fit that failed or isn't converged can't be compared, so stop there
        simple, complex_ = (test, previous) if action == 'remove' else (previous, test)
        if test['problem'] or _chi2_problem(simple['chi2'], complex_['chi2']):
            steps.append((new_pars, None, False))
            break
        # FTest(chi2_1, dof_1, chi2_2, dof_2) compares the simpler model (1) with the more complex model (2)
        ft = FTest(simple['chi2'], simple['dof'], complex_['chi2'], complex_['dof'])
        if action == 'remove':
            passed = _valid_ft(ft) and ft > alpha_remove
        else:
            passed = _valid_ft(ft) and ft <= alpha_add
        steps.append((new_pars, ft, passed))
        if not passed:
            break
        chosen = test['params']
        previous = test
    return chosen, steps


def _describe_steps(prefix, action, steps):
    """ Returns a one-line description of the steps taken through a nested FD or FB series. """
    verb = 'remove' if action == 'remove' else 'add'
    parts = []
    for new_pars, ft, passed in steps:
        result = ("fit failed or not converged" if ft is None else f"F={ft:.3g}") + (f", {verb}" if passed else ", stop")
        parts.append(f"{', '.join(new_pars)} ({result})")
    return f"{prefix} {'removal' if action == 'remove' else 'addition'}, one term at a time: " + "; ".join(parts)


def parseParallelResults(results, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE, verbose=True):
    """
    Checks the results of the F-tests and exports the parameters to be added/removed.

    The following policies are applied: spin frequency derivatives are never recommended to be added, and
    parallax is never recommended to be removed.

    FD and FB parameters each form a series (FD1, FD2, ...), and their tests are nested (e.g. "add FD4" and
    "add FD4 and FD5"; "remove FD2" and "remove FD1 and FD2"). Each series is stepped through one term at a
    time (see `_step_through_nested()`), so the result is the same as re-running the F-tests until nothing
    changes. One consistent decision is made for each series:
        - If adding the next term is significant, the terms that pass are recommended to be added and nothing
          in that series is removed: a significant higher-order term means the lower-order terms are kept.
          Adding takes precedence because it requires much stronger evidence (alpha_add) than removing.
        - Otherwise, terms are recommended to be removed from the top down, until one is significant.

    Input:
    ----------
    results [dict] : Output of `run_Ftests()` or `runFtest_parallel()`.
    alpha_add [float]: F-statistic threshold for adding parameters; added if F <= alpha_add [default: 0.0027].
    alpha_remove [float]: F-statistic threshold for removing parameters; removed if F > alpha_remove [default: 0.1336].
    verbose [boolean]: If True, print the table of F-test results and the FD/FB steps [default: True].

    Returns:
    ---------
    add_params [List of params]: Parameters recommended to be added following the F-tests.
    remove_params [List of params]: Parameters recommended to be removed following the F-tests.
    """
    baseline, tests = _evaluate_tests(results)
    if verbose:
        print("\n".join(_table_lines(baseline, tests, alpha_add=alpha_add, alpha_remove=alpha_remove)))

    n_wrong_way = sum(1 for test in tests if test['problem'] and test['problem'] != "the fit failed")
    if n_wrong_way:
        warnings.warn(f"{n_wrong_way} of {len(tests)} tests have chi2 moving the wrong way (worse with more "
                      "parameters). The baseline fit is probably not converged: refit the model, or increase "
                      "maxiter, before relying on these F-tests.")

    add_params = []
    remove_params = []
    # Nested FD and FB tests, keyed by (prefix, action); these are decided separately below
    nested = {}

    for test in tests:
        action, pars, ft = test['action'], test['params'], test['ft']
        if test['problem']:
            warnings.warn(f"'{action} {', '.join(pars)}' can't be F-tested: {test['problem']}.")
        # Keep nested FD and FB tests aside, including the ones that fail, since they are stepped through below
        prefix = next((pre for pre in ('FD', 'FB') if all(re.fullmatch(pre + r"\d+", p) for p in pars)), None)
        if prefix is not None:
            nested.setdefault((prefix, action), []).append(test)
            continue
        if ft is None:
            continue
        if action == 'remove':
            recommend = _valid_ft(ft) and ft > alpha_remove
        else:
            recommend = _valid_ft(ft) and ft <= alpha_add
        if not recommend:
            continue

        # Policy is to never add additional spin derivatives in general
        if action == 'add' and any(re.fullmatch(r"F\d+", p) for p in pars):
            continue
        # Policy is never to remove parallax
        if action == 'remove' and 'PX' in pars:
            continue
        if action == 'remove':
            remove_params.extend(pars)
        else:
            add_params.extend(pars)

    # Decide each FD and FB series one term at a time. Adding and removing within the same series would give
    # an invalid model (e.g. FD1 and FD3 without FD2), so if a term is added, nothing in that series is removed
    for prefix in ('FD', 'FB'):
        steps_by_action = {}
        chosen_by_action = {}
        for action in ('add', 'remove'):
            chain = sorted(nested.get((prefix, action), []), key=lambda t: len(t['params']))
            chosen_by_action[action], steps_by_action[action] = _step_through_nested(
                chain, action, baseline, alpha_add=alpha_add, alpha_remove=alpha_remove)
        to_add, to_remove = chosen_by_action['add'], chosen_by_action['remove']
        if verbose:
            for action in ('add', 'remove'):
                # Only describe series that go beyond a single step, or that lead to a change
                if len(steps_by_action[action]) > 1 or chosen_by_action[action]:
                    print(_describe_steps(prefix, action, steps_by_action[action]))
        if to_add:
            add_params.extend(to_add)
            if to_remove and verbose:
                print(f"Not recommending removing {', '.join(to_remove)}: adding {', '.join(to_add)} is "
                      f"significant, so the existing {prefix} terms are kept.")
        else:
            remove_params.extend(to_remove)

    return sorted(set(add_params)), sorted(set(remove_params))


def summarize_Ftest(Ftest_dict, fitter=None, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE, alpha=None):
    """
    Function to determine what parameters are recommended to be added/removed from the F-test results.

    Input:
    ----------
    Ftest_dict [dict]: Output of `run_Ftests()` or `runFtest_parallel()`.
    fitter [object]: The PINT fitter object. [currently unused]
    alpha_add [float]: Parameters are recommended to be added if the F-statistic is lower than or equal to
        alpha_add, i.e. they are statistically significant to the timing model [default: 0.0027].
    alpha_remove [float]: Parameters are recommended to be removed if the F-statistic is greater than
        alpha_remove, i.e. they are not statistically significant to the timing model [default: 0.1336].
    alpha [float]: Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns:
    ----------
    add_statement [string]: Statement of which parameters are recommended to be added.
    remove_statement [string]: Statement of which parameters are recommended to be removed.
    """
    if alpha is not None:
        alpha_add = alpha
    add_params, remove_params = parseParallelResults(Ftest_dict, alpha_add=alpha_add, alpha_remove=alpha_remove,
                                                     verbose=False)
    return _recommendation_statements(add_params, remove_params)


def _get_old_ftests(tc):
    """
    Returns the lists of previously recommended parameters to add and remove from the config
    'FTest' entry, or None if there is no 'FTest' entry.
    """
    if 'FTest' not in tc.config:
        return None
    old_ftests = tc.config['FTest'] or {}
    # An empty YAML entry (e.g. "Add:") is read as None
    return list(old_ftests.get('Add') or []), list(old_ftests.get('Remove') or [])


def checkFLogs(add_params, remove_params, tc):
    """
    Compares the F-test recommendations with previous ones stored in the config to avoid
    duplicates and add/remove loops.

    Input:
    --------
    add_params [list]: Parameters recommended to be added.
    remove_params [list]: Parameters recommended to be removed.
    tc [object]: The TimingConfiguration object.

    Returns:
    --------
    add_params [list]: Sorted, unique list of parameters to add.
    remove_params [list]: Sorted, unique list of parameters to remove.
    """
    add_params = sorted(set(add_params))
    remove_params = sorted(set(remove_params))

    old_ftests = _get_old_ftests(tc)
    if old_ftests is None:
        print("\nNo FTest in config.  Continuing...")
        return add_params, remove_params
    old_add, old_remove = old_ftests

    # - Compare old results with new to see if duplicates or loop occur
    for item in add_params:
        if item in old_remove:
            warnings.warn(f'Parameter {item} has already been recommended to be removed.')
    for item in remove_params:
        if item in old_add:
            warnings.warn(f'Parameter {item} has already been recommended to be added.')
    add_params = [item for item in add_params if item not in old_remove]
    remove_params = [item for item in remove_params if item not in old_add]

    return add_params, remove_params


def _process_test_results(results, tc, results_log=False, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE):
    """
    Post-processing shared by `run_Ftests()` and `runFtest_parallel()`: prints the table of F-test results
    and the new recommendations, and prints the combined (new and previous) recommendations in a format to
    copy into the config .yaml file.

    New recommendations that would undo a previous one in the config (e.g. removing a parameter that was
    previously recommended to be added) are dropped with a warning by `checkFLogs()` before they are reported,
    so the printed statements, the .yaml block and the returned lists all agree.

    Input:
    ----------
    results [dict]: Output of the F-test fits, including the 'baseline' fit.
    tc [object]: The TimingConfiguration object.
    results_log [boolean or string]: If True, write the raw results to results.json in the current directory;
        if a string, write them to that path instead [default: False].
    alpha_add [float]: F-statistic threshold for adding parameters [default: 0.0027].
    alpha_remove [float]: F-statistic threshold for removing parameters [default: 0.1336].

    Returns:
    ---------
    ap [list]: Parameters recommended to be added, combined with those already in the config (as in the .yaml block).
    rp [list]: Parameters recommended to be removed, combined with those already in the config (as in the .yaml block).
    """
    if results_log:
        log_file = results_log if isinstance(results_log, (str, os.PathLike)) else "results.json"
        with open(log_file, "w") as fo:
            json.dump(results, fo, indent=4, default=par_fit.astropy_numpy_json_serializer)

    new_ap, new_rp = parseParallelResults(results, alpha_add=alpha_add, alpha_remove=alpha_remove)
    # Drop new recommendations that would undo previous ones in the config
    new_ap, new_rp = checkFLogs(new_ap, new_rp, tc)

    add_statement, remove_statement = _recommendation_statements(new_ap, new_rp)
    print('\n')
    print(add_statement)
    print(remove_statement)

    # Combine with previous FTest results from the config
    old_ftests = _get_old_ftests(tc) or ([], [])
    ap = sorted(set(new_ap + old_ftests[0]))
    rp = sorted(set(new_rp + old_ftests[1]))

    print('\n')
    print('Copy and paste below into .yaml file:')
    print('FTest:\n  Add:', "[" + ",".join(ap) + "]", '\n  Remove:', "[" + ",".join(rp) + "]")
    print('\n')

    return ap, rp


# ---------------------------------------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------------------------------------

def run_Ftests(fitter, tc, results_log=False, maxiter=20, alpha_add=ALPHA_ADD, alpha_remove=ALPHA_REMOVE,
               FDnparams=5, FBnparams=5, alpha=None):
    """
    Runs the F-tests one at a time (serially), for computers that are not set up for parallel processing.
    This runs exactly the same fits as `runFtest_parallel()`, processes the results in the same way, and
    returns the same outputs.

    Input:
    ----------
    fitter [object]: The PINT fitter object. Its current model and TOAs are what is tested.
    tc [object]: The TimingConfiguration object.
    results_log [boolean or string]: If True, write the raw results to results.json in the current directory;
        if a string, write them to that path instead [default: False].
    maxiter [int]: Maximum number of fit iterations for each test [default: 20].
    alpha_add [float]: Parameters are recommended to be added if the F-statistic is lower than or equal to
        alpha_add, i.e. they are statistically significant to the timing model [default: 0.0027].
    alpha_remove [float]: Parameters are recommended to be removed if the F-statistic is greater than
        alpha_remove, i.e. they are not statistically significant to the timing model [default: 0.1336].
    FDnparams [int]: Maximum number of FD parameters to test [default: 5].
    FBnparams [int]: Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].
    alpha [float]: Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns:
    ---------
    results [dict]: Output of the F-test fits, including the 'baseline' fit.
    ap [list]: Parameters recommended to be added, combined with those already in the config (as in the .yaml block).
    rp [list]: Parameters recommended to be removed, combined with those already in the config (as in the .yaml block).
    """
    if alpha is not None:
        alpha_add = alpha
    results = _run_test_fits(fitter, executor=_SerialExecutor(), maxiter=maxiter,
                             FDnparams=FDnparams, FBnparams=FBnparams)
    ap, rp = _process_test_results(results, tc, results_log=results_log,
                                   alpha_add=alpha_add, alpha_remove=alpha_remove)
    return results, ap, rp


def runFtest_parallel(fitter, tc, ncpu=4, results_log=False, maxiter=20, alpha_add=ALPHA_ADD,
                      alpha_remove=ALPHA_REMOVE, FDnparams=5, FBnparams=5):
    """
    Runs the F-tests in parallel with `pint_pal.parallel_fit`, prints the recommendations, and prints
    the combined (new and previous) recommendations in a format to copy into the config .yaml file.
    `run_Ftests()` runs exactly the same fits one at a time and returns the same outputs.

    Input:
    ----------
    fitter [object]: The PINT fitter object. Its current model and TOAs are what is tested.
    tc [object]: The TimingConfiguration object.
    ncpu [int]: Number of CPUs to run the fits on. If 1, the fits are run one at a time exactly as in
        `run_Ftests()`. If None, all available CPUs are used [default: 4].
    results_log [boolean or string]: If True, write the raw results to results.json in the current directory;
        if a string, write them to that path instead [default: False].
    maxiter [int]: Maximum number of fit iterations for each test [default: 20].
    alpha_add [float]: Parameters are recommended to be added if the F-statistic is lower than or equal to
        alpha_add, i.e. they are statistically significant to the timing model [default: 0.0027].
    alpha_remove [float]: Parameters are recommended to be removed if the F-statistic is greater than
        alpha_remove, i.e. they are not statistically significant to the timing model [default: 0.1336].
    FDnparams [int]: Maximum number of FD parameters to test [default: 5].
    FBnparams [int]: Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].

    Returns:
    ---------
    results [dict]: Output of the F-test fits, including the 'baseline' fit.
    ap [list]: Parameters recommended to be added, combined with those already in the config (as in the .yaml block).
    rp [list]: Parameters recommended to be removed, combined with those already in the config (as in the .yaml block).
    """
    if ncpu is not None and ncpu < 1:
        raise ValueError(f"ncpu must be 1 or more (or None to use all CPUs), not {ncpu}.")
    # With one CPU, use the serial executor (as in run_Ftests()) rather than PINT's own single-CPU mode, which
    # reuses one fitter for every test and stops at the first failed fit, so it can give different results
    executor = _SerialExecutor() if ncpu == 1 else None
    results = _run_test_fits(fitter, executor=executor, ncpu=ncpu, maxiter=maxiter,
                             FDnparams=FDnparams, FBnparams=FBnparams)
    ap, rp = _process_test_results(results, tc, results_log=results_log,
                                   alpha_add=alpha_add, alpha_remove=alpha_remove)
    return results, ap, rp
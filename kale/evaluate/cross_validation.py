"""
Functions to do cross-validation with domain adaptation and pre-domain adaptation transformation
for assessing model fitness.
"""

import time
from numbers import Integral, Number, Real
from traceback import format_exc

import numpy as np
from joblib import logger
from sklearn.base import clone, is_classifier
from sklearn.metrics import accuracy_score, check_scoring, get_scorer_names
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import check_cv, LeaveOneGroupOut
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
    _score,
    _warn_or_raise_about_fit_failures,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._array_api import device, get_namespace
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils._param_validation import HasMethods, StrOptions, validate_params
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import _check_method_params, _num_samples, indexable


def _fit_and_score(
    estimator,
    x,
    y,
    transformer,
    domain_adapter,
    factors,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_args,
    score_args,
    return_train_score=False,
    return_parameters=False,
    return_num_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Train the estimator (optionally with a domain adapter) and evaluate its performance.
    The implementation is a modification of the `scikit-learn`'s _fit_and_score function to accomodate
    domain adaptation and pre-domain adaptation transformation. Do not use this function with classification
    labels that has -1 as a label, as it will be treated as an unlabeled sample. This is a limitation of the
    current implementation.

    Args:
        estimator (sklearn.base.BaseEstimator): A scikit-learn estimator implementing fit and predict methods.
        x (array-like): Input data for training and evaluation [num_samples, num_features].
        y (array-like): Target variable for supervised learning [num_samples] or [num_samples, num_targets].
        transformer (sklearn.base.BaseEstimator, optional): An unsupervised transformer implementing fit and transform methods applied before domain adaptation.
        domain_adapter (sklearn.base.BaseEstimator, optional): A domain adapter implementing fit and transform methods.
        factors (array-like, optional): Factors to reduce their influence on the data for domain adaptation [num_samples, num_factors].
        scorer (callable): A scoring function to evaluate the estimator's performance.
        train (array-like): Indices of training samples.
        test (array-like): Indices of testing samples.
        verbose (int): Level of verbosity for logging.
        parameters (dict, optional): Parameters to configure the estimator.
        fit_args (dict, optional): Additional arguments for the estimator's fit method.
        score_args (dict, optional): Additional arguments for the scorer's score method.
        return_train_score (bool): Flag to include training scores in the results.
        return_parameters (bool): Flag to include the estimator's parameters in the results.
        return_num_test_samples (bool): Flag to include the number of test samples in the results.
        return_times (bool): Flag to include fit and score times in the results.
        return_estimator (bool): Flag to include the fitted estimator in the results.
        split_progress (tuple, optional): Progression value for the current split containing `(current_split, total_splits)`.
        candidate_progress (tuple, optional): Progression value for the current candidate containing `(current_candidate, total_candidates)`.
        error_score (float or str): Value to assign to the score if an error occurs during fitting or scoring or to raise error when set to "raise".
    Returns:
        dict: A dictionary containing the results of fitting and scoring, including:
            - "train_scores" (dict, optional): Scores on training set, if `return_train_score=True`.
            - "test_scores" (dict): Scores on testing set.
            - "num_test_samples" (int, optional): Number of test samples, if `return_num_test_samples=True`.
            - "fit_time" (float, optional): Time taken to fit the estimator, if `return_times=True`.
            - "score_time" (float): Time taken to score the estimator.
            - "parameters" (dict, optional): estimator parameters, if `return_parameters=True`.
            - "estimator" (object, optional): The fitted estimator, if `return_estimator=True`.
            - "fit_error" (str, optional): Error message if fitting fails.
    """
    # xp is a drop-in replacement for numpy to be compatible with other libraries
    # like numpy cupy, torch, etc.
    xp, _ = get_namespace(x)
    x_device = device(x)

    train, test = xp.asarray(train, device=x_device), xp.asarray(test, device=x_device)

    if not isinstance(error_score, Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 0:
        lgr = logger.logging.getLogger("fit_and_score")
        lgr.setLevel(logger.logging.INFO)

    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        lgr.info(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    start_time = time.time()

    y_type = type_of_target(y) if y is not None else "unknown"
    x_train, y_train = _safe_split(estimator, x, y, train)
    x_test, y_test = _safe_split(estimator, x, y, test)

    # Adjust length of sample weights
    fit_args = fit_args if fit_args is not None else {}
    fit_args = _check_method_params(x, fit_args, train)
    score_args = score_args if score_args is not None else {}
    score_args_train = _check_method_params(x, score_args, train)
    score_args_test = _check_method_params(x, score_args, test)

    if transformer is not None or domain_adapter is not None:
        x_sampled = xp.concatenate([x_train, x_test], axis=0)
        y_sampled = xp.concatenate([y_train, y_test], axis=0)

        # Mask the labels for the test set to avoid leakage
        if any([y_type.startswith(key) for key in ["binary", "multiclass"]]):
            y_sampled[_num_samples(x_train) - 1 :] = -1
        else:
            raise ValueError("Domain adaptation is only supported for 'binary' or 'multiclass' y.")
        y_sampled = xp.asarray(y_sampled, device=x_device)

    if transformer is not None:
        if parameters is not None:
            keys = [k for k in parameters.keys() if k.startswith("transformer__")]
            transformer_parameters = {k.replace("transformer__", ""): parameters.pop(k) for k in keys}
            transformer.set_params(**clone(transformer_parameters, safe=False))

        # For now, only unsupervised transformers are supported
        transformer.fit(x_sampled)
        x_sampled = transformer.transform(x_sampled)

    if domain_adapter is not None:
        factors_sampled, factors_train, factors_test = [None] * 3
        if factors is not None:
            factors_train = _safe_indexing(factors, train)
            factors_test = _safe_indexing(factors, test)
            factors_sampled = xp.concatenate([factors_train, factors_test], axis=0)

        if parameters is not None:
            keys = [k for k in parameters.keys() if k.startswith("domain_adapter__")]
            da_parameters = {k.replace("domain_adapter__", ""): parameters.pop(k) for k in keys}
            domain_adapter.set_params(**clone(da_parameters, safe=False))

        domain_adapter.fit(x_sampled, y_sampled, factors_sampled)
        x_train = domain_adapter.transform(x_train, factors_train)
        x_test = domain_adapter.transform(x_test, factors_test)

    if y is not None and any([y_type.startswith(key) for key in ["binary", "multiclass"]]):
        train_labeled = y_train != -1
        train = _safe_indexing(train, train_labeled)
        x_train = _safe_indexing(x_train, train_labeled)
        y_train = _safe_indexing(y_train, train_labeled)
        fit_args = _check_method_params(x_train, fit_args, train_labeled)

    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))

    result = {}
    try:
        if y_train is None:
            estimator.fit(x_train, **fit_args)
        else:
            estimator.fit(x_train, y_train, **fit_args)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, Number):
            if isinstance(scorer, _MultimetricScorer):
                test_scores = {name: error_score for name in scorer._scorers}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, x_test, y_test, scorer, score_args_test, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, x_train, y_train, scorer, score_args_train, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        lgr.info(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_num_test_samples:
        result["num_test_samples"] = _num_samples(x_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def leave_one_group_out(x, y, groups, estimator, use_domain_adaptation=False) -> dict:
    """
    Perform leave one group out cross validation for a given estimator.

    Args:
        x (np.ndarray or torch.tensor): Input data [num_samples, num_features].
        y (np.ndarray or torch.tensor): Target labels [num_samples].
        groups (np.ndarray or torch.tensor): Group labels to be left out [num_samples].
        estimator (estimator object): Machine learning estimator to be evaluated from kale or scikit-learn.
        use_domain_adaptation (bool): Whether to use domain adaptation, i.e., leveraging test data, during training.

    Returns:
        dict: A dictionary containing results for each target group with 3 keys.
            - 'Target': A list of unique target groups or classes. The final entry is "Average".
            - 'Num_samples': A list where each entry indicates the number of samples in its corresponding target group.
                            The final entry represents the total number of samples.
            - 'Accuracy': A list where each entry indicates the accuracy score for its corresponding target group.
                            The final entry represents the overall mean accuracy.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    group_mat = enc.fit_transform(groups.reshape(-1, 1)).toarray()
    target, num_samples, accuracy = np.unique(groups).tolist(), [], []
    y_pred = np.zeros(y.shape)  # Store all predicted labels to compute accuracy

    for train, test in LeaveOneGroupOut().split(x, y, groups=groups):
        x_src, x_tgt = x[train], x[test]
        y_src, y_tgt = y[train], y[test]

        if use_domain_adaptation:
            estimator.fit(np.concatenate((x_src, x_tgt)), y_src, np.concatenate((group_mat[train], group_mat[test])))
        else:
            estimator.fit(x_src, y_src)

        y_pred[test] = estimator.predict(x_tgt)
        num_samples.append(x_tgt.shape[0])
        accuracy.append(accuracy_score(y_tgt, y_pred[test]))

    target.append("Average")
    num_samples.append(x.shape[0])
    accuracy.append(accuracy_score(y, y_pred))

    return {
        "Target": target,
        "Num_samples": num_samples,
        "Accuracy": accuracy,
    }


@validate_params(
    {
        "estimator": [HasMethods(["fit", "predict", "score"])],
        "x": ["array-like", "sparse matrix"],
        "y": ["array-like", None],
        "groups": ["array-like", None],
        "transformer": [HasMethods(["fit", "transform"]), None],
        "domain_adapter": [HasMethods(["fit", "transform"]), None],
        "factors": ["array-like", "sparse matrix", None],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            list,
            tuple,
            dict,
            None,
        ],
        "cv": ["cv_object"],
        "num_jobs": [Integral, None],
        "verbose": ["verbose"],
        "params": [dict, None],
        "pre_dispatch": [Integral, str],
        "return_train_score": ["boolean"],
        "return_estimator": ["boolean"],
        "return_indices": ["boolean"],
        "error_score": [StrOptions({"raise"}), Real],
    },
    prefer_skip_nested_validation=False,  # estimator is not validated yet
)
def cross_validate(
    estimator,
    x,
    y=None,
    groups=None,
    transformer=None,
    domain_adapter=None,
    factors=None,
    scoring=None,
    cv=None,
    num_jobs=None,
    verbose=0,
    parameters=None,
    fit_args=None,
    score_args=None,
    pre_dispatch="2*num_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    error_score=np.nan,
):
    """Run cross-validation and record fit and score times.

    Args:
        estimator (sklearn.base.BaseEstimator): A scikit-learn estimator implementing fit and predict methods.
        x (array-like): Input data for training and evaluation [num_samples, num_features].
        y (array-like): Target variable for supervised learning [num_samples] or [num_samples, num_targets].
        groups (array-like, optional): Group labels for the samples used while splitting the dataset into train/test sets.
        transformer (sklearn.base.BaseEstimator, optional): An unsupervised transformer implementing fit and transform methods applied before domain adaptation.
        domain_adapter (sklearn.base.BaseEstimator, optional): A domain adapter implementing fit and transform methods.
        factors (array-like, optional): Factors to reduce their influence on the data during domain adaptation [num_samples, num_factors].
        scoring (callable, list, tuple, dict, optional): A scoring function or a list of scoring functions to evaluate the estimator's performance.
        cv (cv_object, optional): Cross-validation splitting strategy.
        num_jobs (int, optional): Number of jobs to run in parallel.
        verbose (int): Level of verbosity for logging.
        parameters (dict, optional): Parameters to configure the estimator.
        fit_args (dict, optional): Additional arguments for the estimator's fit method.
        score_args (dict, optional): Additional arguments for the scorer's score method.
        pre_dispatch (int, str): Controls the number of jobs that get dispatched during parallel execution.
        return_train_score (bool): Whether to include training scores in the results.
        return_estimator (bool): Whether to include the fitted estimator in the results.
        return_indices (bool): Whether to include the indices of the training and testing sets in the results.
        error_score (float or str): Value to assign to the score if an error occurs during fitting or scoring or to raise error when set to "raise".
    Returns:
        dict: A dictionary containing the results of fitting and scoring, including:
            - "train_scores" (dict, optional): Scores on training set, if `return_train_score=True`.
            - "test_scores" (dict): Scores on testing set.
            - "fit_time" (float, optional): Time taken to fit the estimator, if `return_times=True`.
            - "score_time" (float): Time taken to score the estimator.
            - "estimator" (object, optional): The fitted estimator, if `return_estimator=True`.
            - "indices" (dict, optional): Indices of the training and testing sets, if `return_indices=True`.
    """
    x, y, groups, factors = indexable(x, y, groups, factors)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    parameters = {} if parameters is None else parameters
    fit_args = {} if fit_args is None else fit_args
    score_args = {} if score_args is None else score_args

    scorers = check_scoring(estimator, scoring, raise_exc=(error_score == "raise"))

    indices = cv.split(x, y, groups)
    if return_indices:
        indices = list(indices)

    parallel = Parallel(num_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            x,
            y,
            transformer=clone(transformer) if transformer else None,
            domain_adapter=clone(domain_adapter) if domain_adapter else None,
            factors=factors,
            scorer=scorers,
            train=train,
            test=test,
            verbose=verbose,
            parameters=parameters,
            fit_args=fit_args,
            score_args=score_args,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
        )
        for train, test in indices
    )

    _warn_or_raise_about_fit_failures(results, error_score)

    # For callable scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    aggregated_results = {}
    aggregated_results["fit_time"] = results["fit_time"]
    aggregated_results["score_time"] = results["score_time"]

    if return_estimator:
        aggregated_results["estimator"] = results["estimator"]

    if return_indices:
        aggregated_results["indices"] = {}
        aggregated_results["indices"]["train"], aggregated_results["indices"]["test"] = zip(*indices)

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        aggregated_results["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            aggregated_results[key] = train_scores_dict[name]

    return aggregated_results

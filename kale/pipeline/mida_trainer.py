# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
# =============================================================================

"""Implementation of Transformer -> Maximum independence domain adaptation -> Estimator Pipeline

References:
    [1] Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace using domain features and
        independence maximization. IEEE transactions on cybernetics, 48(1), pp.288-299.
"""
import logging
import time
from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.base import _fit_context, clone, is_classifier
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import check_cv, ParameterGrid, ParameterSampler
from sklearn.model_selection._search import _insert_error_scores, _warn_or_raise_about_fit_failures, BaseSearchCV
from sklearn.utils._param_validation import HasMethods, Integral, Interval, StrOptions
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import _check_method_params, check_is_fitted, indexable

from ..embed.factorization import MIDA
from ..evaluate.cross_validation import _fit_and_score


class MIDATrainer(BaseSearchCV):
    """Trainer for estimator with maximum independence domain adaptation (MIDA) incorporated.
    The pipeline has a processing order of: `transformer` (optional) -> `MIDA` -> `estimator`.
    The `transformer` must be an unsupervised ones, like `StandardScaler` or `PCA`,
    as its fit method is called on the merged training and test data. To set parameters
    for both `transformer` and `MIDA`, use the prefixes `transformer__` and `domain_adapter__`, respectively.

    Args:
        estimator (sklearn.base.BaseEstimator): The base estimator to be trained.
        param_grid (dict or list): The parameter grid to search over. Use 'transformer__' and 'domain_adapter__' as prefixes for the transformer and MIDA parameters, respectively.
        transformer (sklearn.base.BaseEstimator, optional): The transformer to be applied before MIDA. Default is None.
        search_strategy ("grid" or "random"): The search strategy to use. Can be "grid" or "random". Default is "grid".
        num_iter (int): The number of iterations for random search. Default is None.
        scoring (str or callable or list or tuple or dict, optional): The scoring metric(s) to use. Default is None.
        n_jobs (int): The number of jobs to run in parallel for joblib.Parallel. Default is None.
        pre_dispatch (int or str): Controls the number of jobs that get dispatched during parallel execution. Default is "2*n_jobs".
        refit (bool): Whether to refit the best estimator. Default is True.
        cv (int or cv-object or iterable): The cross-validation splitting strategy. Default is None.
        verbose (int): Controls the verbosity of the output. Default is 0.
        random_state (int or np.random.RandomState, optional): Controls the randomness of the estimator. Default is None.
        error_score ('raise' or numeric): Value to assign to the score if an error occurs. Default is np.nan.
        return_train_score (bool): Whether to include training scores in the results. Default is False.
    Examples:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from kale.pipeline.mida_trainer import MIDATrainer
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.metrics import accuracy_score
        >>> # Generate synthetic data
        >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        >>> # Generate domain labels
        >>> factors = np.random.randint(0, 2, size=(X_train.shape[0], 1))
        >>> # Define the base estimator and parameter grid
        >>> estimator = RandomForestClassifier(random_state=42)
        >>> param_grid = {
                "num_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "transformer__with_mean": [True, False],
                "domain_adapter__num_components": [2, 5],
            }
        >>> transformer = StandardScaler()
        >>> trainer = MIDATrainer(
                estimator=estimator,
                param_grid=param_grid,
                transformer=transformer,
                search_strategy="grid",
                scoring="accuracy",
                n_jobs=-1,
                cv=5,
                verbose=1,
            )
        >>> # Fit the trainer
        >>> trainer.fit(X, y, factors=factors)
        >>> # Get cross-validation results
        >>> cv_results = trainer.cv_results_
    """

    _parameter_constraints = {
        **BaseSearchCV._parameter_constraints,
        "use_mida": ["boolean"],
        "param_grid": [dict, list],
        "transformer": [HasMethods(["fit", "transform"]), None],
        "search_strategy": [StrOptions({"grid", "random"})],
        "num_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_grid,
        use_mida=True,
        transformer=None,
        search_strategy="grid",
        num_iter=10,
        scoring=None,
        # n_jobs can't be changed to num_jobs since it is used in the parent class
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        refit=True,
        cv=None,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.use_mida = use_mida
        self.transformer = transformer
        self.param_grid = param_grid
        self.search_strategy = search_strategy
        self.num_iter = num_iter
        self.random_state = random_state

    def adapt(self, x, factors=None):
        """Adapt the estimator to the given data.

        Args:
            x (array-like): The input data.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The transformed or adapted data.
        """
        check_is_fitted(self)
        if self.transformer is not None:
            x = self.best_transformer_.transform(x)

        if self.use_mida:
            x = self.best_mida_.transform(x, factors)

        return x

    def score(self, x, y=None, factors=None, **params):
        """Compute the score of the estimator on the given data.

        Args:
            x (array-like): The input data.
            y (array-like): The target values.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
            **params: Additional parameters for the estimator.
        Returns:
            float: The score of the estimator.
        """
        check_is_fitted(self)
        x = self.adapt(x, factors)
        return super().score(x, y, **params)

    def score_samples(self, x, factors=None):
        """Compute the log-likelihood of the samples.

        Args:
            x (array-like): The input data.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The log-likelihood of the samples.
        """
        x = self.adapt(x, factors)
        return super().score_samples(x)

    def predict(self, x, factors=None):
        """Predict using the best pipeline.

        Args:
            x (array-like): The input data.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The predicted target.
        """
        x = self.adapt(x, factors)
        return super().predict(x)

    def predict_proba(self, x, factors=None):
        """Predict class probabilities using the best pipeline.

        Args:
            x (array-like): The input data.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The predicted class probabilities.
        """
        x = self.adapt(x, factors)
        return super().predict_proba(x)

    def predict_log_proba(self, x, factors=None):
        """Predict log class probabilities using the best pipeline.

        Args:
            x (array-like): The input data.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The predicted log class probabilities.
        """
        x = self.adapt(x, factors)
        return super().predict_log_proba(x)

    def decision_function(self, x, factors=None):
        """Compute the decision function using the best pipeline."""
        x = self.adapt(x, factors)
        return super().decision_function(x)

    def transform(self, x, factors=None):
        """Transform the data using the best pipeline."""
        x = self.adapt(x, factors)
        return super().transform(x)

    def inverse_transform(self, x):
        """Inverse transform the data using the best pipeline.
        Note that this method is not compatible for MIDA when `augment=True`.

        Args:
            x (array-like): The input data.
        Returns:
            array-like: The inverse transformed data.
        """
        check_is_fitted(self)
        x = super().inverse_transform(x)

        if hasattr(self.best_mida_, "inverse_transform"):
            x = self.best_mida_.inverse_transform(x)

        if self.transformer is not None and hasattr(self.best_transformer_, "inverse_transform"):
            x = self.best_transformer_.inverse_transform(x)

        return x

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""

        # Can't be replace since it is used to validate the input data
        # by scikit-learn, default across all estimators

        # Trick to call the try exception block of the parent class
        super().n_features_in_
        if self.transformer is not None:
            return self.best_transformer_.n_features_in_
        return self.best_mida_.n_features_in_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y=None, factors=None, **params):
        """Fit and tune the pipeline with the given data.

        Args:
            x (array-like): The input data.
            y (array-like): The target values.
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
            **params: Additional parameters for the estimator.
        Returns:
            self: The fitted trainer.
        """

        estimator = self.estimator
        transformer = self.transformer
        mida = MIDA() if self.use_mida else None
        scorers, refit_metric = self._get_scorers()

        x, y, factors = indexable(x, y, factors)
        params = _check_method_params(x, params)
        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(x, y, **routed_params.splitter.split)

        base_estimator = clone(estimator)
        base_transformer = clone(transformer) if transformer else None
        base_mida = clone(mida) if self.use_mida else None

        parallel = Parallel(self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_args=routed_params.estimator.fit,
            score_args=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_num_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    logger = logging.getLogger("MIDATrainer.fit")
                    logger.setLevel(logging.INFO)
                    logger.info(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(n_splits, n_candidates, n_candidates * n_splits)
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        x,
                        y,
                        transformer=base_transformer,
                        domain_adapter=base_mida,
                        factors=factors,
                        train=train,
                        test=test,
                        parameters=clone(parameters, safe=False),
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),
                        enumerate(cv.split(x, y, **routed_params.splitter.split)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. " "Was the CV iterator empty? " "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(all_candidate_params, n_splits, all_out, all_more_results)

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callable scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786

            # Temporary workaround to make domain adaptation work.
            # The ideal scenario is to allow set params to work with all
            # estimators simultaneously
            best_params = clone(self.best_params_, safe=False)

            transformer_keys = [k for k in best_params if k.startswith("transformer__")]
            best_transformer_params = {k.replace("transformer__", ""): best_params.pop(k) for k in transformer_keys}

            mida_keys = [k for k in best_params if k.startswith("domain_adapter__")]
            best_mida_params = {k.replace("domain_adapter__", ""): best_params.pop(k) for k in mida_keys}

            if base_transformer is not None:
                self.best_transformer_ = clone(base_transformer)
                self.best_transformer_.set_params(**clone(best_transformer_params, safe=False))

            if self.use_mida:
                self.best_mida_ = clone(base_mida)
                self.best_mida_.set_params(**clone(best_mida_params, safe=False))

            self.best_estimator_ = clone(base_estimator)
            self.best_estimator_.set_params(**clone(best_params, safe=False))

            refit_start_time = time.time()

            if self.transformer is not None:
                x = self.best_transformer_.fit_transform(x)

            if y is not None and self.use_mida:
                x = self.best_mida_.fit_transform(x, y, factors)
            elif self.use_mida:
                x = self.best_mida_.fit_transform(x, factors=factors)

            if y is not None:
                self.best_estimator_.fit(x, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(x, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        # n_splits_ is a parent attribute within BaseSearchCV
        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _run_search(self, evaluate_candidates):
        """Search hyperparameter candidates from param_distributions

        Args:
            evaluate_candidates (callable): Function to evaluate candidates.
        """
        if self.search_strategy == "grid":
            evaluate_candidates(ParameterGrid(self.param_grid))
            return

        evaluate_candidates(ParameterSampler(self.param_grid, self.num_iter, random_state=self.random_state))

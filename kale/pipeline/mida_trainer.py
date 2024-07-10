# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
# =============================================================================

"""Implementation of Transformer -> Maximum independence domain adaptation -> Estimator Pipeline

References:
    [1] Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace using domain features and
        independence maximization. IEEE transactions on cybernetics, 48(1), pp.288-299.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid
from sklearn.preprocessing import FunctionTransformer

from ..embed.factorization import MIDA


class MIDATrainer(BaseEstimator, ClassifierMixin):
    """Trainer of pipeline: Transformer (optional) -> Maximum independence domain adaptation -> Estimator

    Args:
        estimator (BaseEstimator): Estimator object implementing 'fit'
        estimator_param_grid (dict, optional): Grids for searching the optimal hyperparameters of the estimator.
            Defaults to None.
        transformer (BaseEstimator, optional): Transformer object implementing 'fit_transform'. Defaults to None.
        mida_param_grid (dict, optional): Grids for searching the optimal hyperparameters of MIDA. Defaults to None.
        transformer_param_grid (dict, optional): Grids for searching the optimal hyperparameters of the transformer.
            Defaults to None.
    """

    def __init__(
        self,
        estimator,
        estimator_param_grid=None,
        transformer=None,
        mida_param_grid=None,
        transformer_param_grid=None,
    ):
        self.estimator = estimator
        self.transformer = transformer
        self.mida_param_grid = mida_param_grid
        self.transformer_param_grid = transformer_param_grid
        self.estimator_param_grid = estimator_param_grid
        self.best_transformer = None
        self.best_transformer_params = None
        self.best_mida = None
        self.best_mida_params = None
        self.best_estimator = None
        self.best_estimator_params = None
        self.best_score = None

    def fit(self, x, y, groups=None):
        """Fit a Transformer -> Maximum independence domain adaptation -> Estimator pipeline with the given data x ,
            labels y, and group labels groups

        Args:
            x (array-like): Training input data matrix, shape (n_samples, n_features)
            y (array-like): Labels, shape (n_labeled_samples,)
            groups (array-like, optional): Domain/group labels of x, shape (n_samples,)

        Returns:
            self
        """
        if self.transformer is None:
            self.transformer = FunctionTransformer()
        if self.transformer_param_grid is None:
            self.transformer_param_grid = {key: [val] for key, val in self.transformer.get_params().items()}
        transformer_grid = ParameterGrid(self.transformer_param_grid)
        for transformer_params in transformer_grid:
            self.transformer.set_params(**transformer_params)
            x_transformed = self.transformer.fit_transform(x)
            self._fit_mida(x_transformed, y, groups, transformer_params)

        self.best_transformer = self.transformer.set_params(**self.best_transformer_params)
        self.best_mida = MIDA(**self.best_mida_params)
        self.best_estimator = self.estimator.set_params(**self.best_estimator_params)

        x_transformed = self.best_transformer.transform(x)
        x_transformed = self.best_mida.fit_transform(x_transformed, y=y, groups=groups)
        self.best_estimator.fit(x_transformed[y.shape[0] :,], y)

        return self

    def _fit_mida(self, x_transformed, y, groups, transformer_params):
        """Fit MIDA and estimator with the given data x_transformed and labels

        Args:
            x_transformed (array-like): Transformed data, shape (n_samples, n_features)
            y (array-like): Data labels, shape (n_labeled_samples,)
            groups (array-like): Group labels for the samples, shape (n_samples,)
            transformer_params (dict): Parameters of the transformer

        Returns:
            self
        """
        self.best_mida = MIDA()
        if self.mida_param_grid is None:
            self.mida_param_grid = {key: [val] for key, val in self.best_mida.get_params().items()}
        mida_grid = ParameterGrid(self.mida_param_grid)
        for mida_params in mida_grid:
            mida = MIDA(**mida_params)
            x_transformed = mida.fit_transform(x_transformed, y=y, groups=groups)
            cv = 5
            if np.unique(groups).shape[0] > 2:
                cv = LeaveOneGroupOut().split(x_transformed[y.shape[0] :], y, groups[y.shape[0] :])
            self._fit_estimator(x_transformed, y, mida_params, transformer_params, cv=cv)

        return self

    def _fit_estimator(self, x_transformed, y, mida_params, transformer_params, cv=5):
        """Fit the estimator with the given data x_transformed and labels y

        Args:
            x_transformed (array-like): Transformed data, shape (n_samples, n_features)
            y (array-like): Data labels, shape (n_labeled_samples,)
            mida_params (dict): Hyperparameters of the MIDA
            transformer_params (dict): Hyperparameters of the transformer

        Returns:
            self
        """
        if self.estimator_param_grid is None:
            self.estimator_param_grid = {key: [val] for key, val in self.estimator.get_params().items()}

        clf = GridSearchCV(self.estimator, self.estimator_param_grid, cv=cv)
        clf.fit(x_transformed[y.shape[0] :,], y)
        if self.best_score is None or clf.best_score_ > self.best_score:
            self.best_score = clf.best_score_
            self.best_estimator_params = clf.best_params_
            self.best_mida_params = mida_params
            self.best_transformer_params = transformer_params

        return self

    def predict(self, x, groups=None):
        """Predict the labels for the given data x

        Args:
            x (array-like tensor): The data matrix for which we want to get the predictions,
                shape (n_samples, n_features).
            groups (array-like, optional): Group labels of x, shape (n_samples,)

        Returns:
            array-like: Predicted labels, shape (n_samples,)
        """
        return self.best_estimator.predict(self.transform(x, groups=groups))

    def decision_function(self, x, groups=None):
        """Decision scores of each class for the given data x

        Args:
            x (array-like tensor): The data matrix for which we want to get the decision scores,
                shape (n_samples, n_features).
            groups (array-like, optional): Group labels for the samples, shape (n_samples,)

        Returns:
            array-like: Decision scores, shape (n_samples,) for binary case, else (n_samples, n_class)
        """

        return self.best_estimator.decision_function(self.transform(x, groups=groups))

    def predict_proba(self, x, groups=None):
        """Probability of each class for the given data x. Not supported by "linear_svc".

        Args:
            x (array-like tensor): The data matrix for which we want to get the predictions probabilities,
                shape (n_samples, n_features)
            groups (array-like, optional): Group labels of x, shape (n_samples,)

        Returns:
            array-like: Prediction probabilities, shape (n_samples, n_class)
        """
        return self.best_estimator.predict_proba(self.transform(x, groups=groups))

    def transform(self, x, groups=None):
        """Transform the input data x

        Args:
            x (array-like tensor): Data matrix to get the transformed data, shape (n_samples, n_features)
            groups (array-like, optional): Group labels of x, shape (n_samples,)

        Returns:
            array-like: Transformed data, shape (n_samples, n_features)
        """

        return self.best_mida.transform(self.best_transformer.transform(x), groups=groups)

    def fit_predict(self, x, y, groups=None):
        """Fit a pipeline with the given data x, labels y, and group labels groups, and predict the labels for x

        Args:
            x (array-like): Training input data matrix, shape (n_samples, n_features)
            y (array-like): Labels, shape (n_labeled_samples,)
            groups (array-like, optional): Domain/group labels of x, shape (n_samples,)

        Returns:
            array-like: Predicted labels, shape (n_samples,)
        """
        self.fit(x, y, groups)
        return self.predict(x, groups=groups)

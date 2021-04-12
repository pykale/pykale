# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Implementation of MPCA->Feature Selection->Linear SVM/LogisticRegression Pipeline

References:
    [1] Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2020). A machine
    learning cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial
    hypertension diagnosis. European Heart Journal-Cardiovascular Imaging.
    [2] Song, X., Meng, L., Shi, Q., & Lu, H. (2015, October). Learning tensor-based features for whole-brain fMRI
    classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention
    (pp. 613-620). Springer, Cham.
    [3] Lu, H., Plataniotis, K. N., & Venetsanopoulos, A. N. (2008). MPCA: Multilinear principal component analysis of
    tensor objects. IEEE Transactions on Neural Networks, 19(1), 18-39.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted

from ..embed.mpca import MPCA

param_c_grids = list(np.logspace(-4, 2, 7))
classifiers = {
    "svc": [SVC, {"kernel": ["linear"], "C": param_c_grids}],
    "linear_svc": [LinearSVC, {"C": param_c_grids}],
    "lr": [LogisticRegression, {"C": param_c_grids}],
}

# k-fold cross validation used for grid search, i.e. searching for optimal value of C
default_search_params = {"cv": 5}
default_mpca_params = {"var_ratio": 0.97, "return_vector": True}


class MPCATrainer(BaseEstimator, ClassifierMixin):
    def __init__(
        self, classifier="svc", classifier_params="auto", mpca_params=None, n_features=None, search_params=None
    ):
        """Trainer of pipeline: MPCA->Feature selection->Classifier

        Args:
            classifier (str, optional): Available classifier options: {"svc", "linear_svc", "lr"}, where "svc" trains a
                support vector classifier, supports both linear and non-linear kernels, optimizes with library "libsvm";
                "linear_svc" trains a support vector classifier with linear kernel only, and optimizes with library
                "liblinear", which suppose to be faster and better in handling large number of samples; and "lr" trains
                a classifier with logistic regression. Defaults to "svc".
            classifier_params (dict, optional): Parameters of classifier. Defaults to 'auto'.
            mpca_params (dict, optional): Parameters of MPCA. Defaults to None.
            n_features (int, optional): Number of features for feature selection. Defaults to None, i.e. all features
                after dimension reduction will be used.
            search_params (dict, optional): Parameters of grid search. Defaults to None.

        """
        if classifier not in ["svc", "linear_svc", "lr"]:
            error_msg = "Valid classifier should be 'svc', 'linear_svc', or 'lr', but given %s" % classifier
            logging.error(error_msg)
            raise ValueError(error_msg)

        self.classifier = classifier
        # init mpca object
        if mpca_params is None:
            self.mpca_params = default_mpca_params
        else:
            self.mpca_params = mpca_params
        self.mpca = MPCA(**self.mpca_params)
        # init feature selection parameters
        self.n_features = n_features
        self.feature_order = None
        # init classifier object
        if search_params is None:
            self.search_params = default_search_params
        else:
            self.search_params = search_params

        self.auto_classifier_param = False
        if classifier_params == "auto":
            self.auto_classifier_param = True
            clf_param_grid = classifiers[classifier][1]
            self.grid_search = GridSearchCV(
                classifiers[classifier][0](), param_grid=clf_param_grid, **self.search_params
            )
            self.clf = None
        elif isinstance(classifier_params, dict):
            self.clf = classifiers[classifier][0](**classifier_params)
        else:
            error_msg = "Invalid classifier parameter type"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def fit(self, x, y):
        """Fit a pipeline with the given data x and labels y

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)
            y (array-like): data labels, shape (n_samples, )

        Returns:
            self
        """
        # fit mpca
        self.mpca.fit(x)
        self.mpca.set_params(**{"return_vector": True})
        x_transformed = self.mpca.transform(x)

        # feature selection
        if self.n_features is None:
            self.n_features = x_transformed.shape[1]
            self.feature_order = self.mpca.idx_order
        else:
            f_score, p_val = f_classif(x_transformed, y)
            self.feature_order = (-1 * f_score).argsort()
        x_transformed = x_transformed[:, self.feature_order][:, : self.n_features]

        # fit classifier
        if self.auto_classifier_param:
            self.grid_search.param_grid["C"].append(1 / x.shape[0])
            self.grid_search.fit(x_transformed, y)
            self.clf = self.grid_search.best_estimator_
        if self.classifier == "svc":
            self.clf.set_params(**{"probability": True})

        self.clf.fit(x_transformed, y)

    def predict(self, x):
        """Predict the labels for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: Predicted labels, shape (n_samples, )
        """
        return self.clf.predict(self._extract_feature(x))

    def decision_function(self, x):
        """Decision scores of each class for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: decision scores, shape (n_samples,) for binary case, else (n_samples, n_class)
        """
        return self.clf.decision_function(self._extract_feature(x))

    def predict_proba(self, x):
        """Probability of each class for the given data x. Not supported by "linear_svc".

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: probabilities, shape (n_samples, n_class)
        """
        if self.classifier == "linear_svc":
            error_msg = "Linear SVC does not support computing probability."
            logging.error(error_msg)
            raise ValueError(error_msg)
        return self.clf.predict_proba(self._extract_feature(x))

    def _extract_feature(self, x):
        """Extracting features for the given data x with MPCA->Feature selection

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: n_new, shape (n_samples, n_features)
        """
        check_is_fitted(self.clf)
        x_transformed = self.mpca.transform(x)

        return x_transformed[:, self.feature_order][:, : self.n_features]

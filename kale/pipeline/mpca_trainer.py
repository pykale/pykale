# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Implementation of MPCA->Feature Selection->Linear SVM/LogisticRegression Pipeline
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from ..embed.mpca import MPCA

classifiers = {"svc": [SVC, {"kernel": ["linear"], "C": np.logspace(-3, 2, 6)}],
               "lr": [LogisticRegression, {"C": np.logspace(-3, 2, 6)}]}

default_search_params = {'cv': 5}
default_mpca_params = {"var_ratio": 0.97, "return_vector": True}


class MPCATrainer(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier='svc', classifier_params='auto', mpca_params=None,
                 n_features=None, search_params=None):
        """Trainer of pipeline: MPCA->Feature selection->Classifier

        Args:
            classifier (str, optional): Classifier for training. Options: support vector machine (svc) or
                logistic regression (lr). Defaults to 'svc'.
            classifier_params (dict, optional): Parameters of classifier. Defaults to 'auto'.
            mpca_params (dict, optional): Parameters of Multi-linear PCA. Defaults to None.
            n_features (int, optional): Number of features for feature selection. Defaults to None.
            search_params (dict, optional): Parameters of grid search. Defaults to None.

        """
        if classifier not in ['svc', 'lr']:
            error_msg = "Valid classifier should be 'svc' or 'lr', but given %s" % classifier
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
            clf_param_gird = classifiers[classifier][1]
            self.grid_search = GridSearchCV(classifiers[classifier][0](),
                                            param_grid=clf_param_gird,
                                            **self.search_params)
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
        x_proj = self.mpca.transform(x)

        # feature selection
        if self.n_features is None:
            self.n_features = x_proj.shape[0]
            self.feature_order = self.mpca.idx_order
        else:
            f_score, p_val = f_classif(x_proj, y)
            self.feature_order = (-1 * f_score).argsort()
        x_proj = x_proj[:, self.feature_order][:, :self.n_features]

        # fit classifier
        if self.auto_classifier_param:
            self.grid_search.fit(x_proj, y)
            self.clf = self.grid_search.best_estimator_
        if self.classifier == "svc":
            self.clf.set_params(**{"probability": True})

        self.clf.fit(x_proj, y)

    def predict(self, x):
        """Predict the labels for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: Predicted labels, shape (n_samples, )
        """
        return self.clf.predict(self.feature_extract(x))

    def decision_function(self, x):
        """Decision scores of each class for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: decision scores, shape (n_samples, n_class)
        """
        return self.clf.decision_function(self.feature_extract(x))

    def predict_proba(self, x):
        """Probability of each class for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: probabilities, shape (n_samples, n_class)
        """
        return self.clf.predict_proba(self.feature_extract(x))

    def feature_extract(self, x):
        """Extracting features for the given data x

        Args:
            x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: n_new, shape (n_samples, n_features)
        """
        check_is_fitted(self.clf)

        x_proj = self.mpca.transform(x)
        x_new = x_proj[:, self.feature_order][:, :self.n_features]

        return x_new

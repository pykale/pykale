# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk/sz144@outlook.com
# =============================================================================
"""Multi-source domain adaptation pipelines
"""

import logging
import time
from collections import defaultdict
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import (
    _fit_context,
    BaseEstimator,
    ClassifierMixin,
    clone,
    is_classifier,
    MetaEstimatorMixin,
    TransformerMixin,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import check_cv, ParameterGrid, ParameterSampler
from sklearn.model_selection._search import _insert_error_scores, _warn_or_raise_about_fit_failures, BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC, SVC
from sklearn.utils._param_validation import HasMethods, Integral, Interval, StrOptions
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils.validation import _check_method_params, check_is_fitted, indexable
from torch.linalg import multi_dot
from torch.nn.functional import one_hot

import kale.evaluate.metrics as losses
from kale.embed.image_cnn import _Bottleneck
from kale.pipeline.domain_adapter import BaseAdaptTrainer

from ..embed.factorization import MIDA
from ..evaluate.cross_validation import _fit_and_score


def create_ms_adapt_trainer(method: str, dataset, feature_extractor, task_classifier, **train_params):
    """Methods for multi-source domain adaptation

    Args:
        method (str): Multi-source domain adaptation method, M3SDA or MFSAN
        dataset (kale.loaddata.multi_domain.MultiDomainAdapDataset): the multi-domain datasets to be used for train,
            validation, and tests.
        feature_extractor (torch.nn.Module): feature extractor network
        task_classifier (torch.nn.Module): task classifier network

    Returns:
        [pl.LightningModule]: Multi-source domain adaptation trainer.
    """
    method_dict = {"M3SDA": M3SDATrainer, "DIN": _DINTrainer, "MFSAN": MFSANTrainer}
    method = method.upper()
    if method not in method_dict.keys():
        raise ValueError("Unsupported multi-source domain adaptation methods %s" % method)
    else:
        return method_dict[method](dataset, feature_extractor, task_classifier, **train_params)


def _average_cls_output(x, classifiers: nn.ModuleDict):
    cls_output = [classifiers[key](x) for key in classifiers]

    return torch.stack(cls_output).mean(0)


class BaseMultiSourceTrainer(BaseAdaptTrainer):
    """Base class for all domain adaptation architectures

    Args:
        dataset (kale.loaddata.multi_domain): the multi-domain datasets to be used for train, validation, and tests.
        feature_extractor (torch.nn.Module): the feature extractor network
        task_classifier (torch.nn.Module): the task classifier network
        n_classes (int): number of classes
        target_domain (str): target domain name
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        n_classes: int,
        target_domain: str,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)
        self.n_classes = n_classes
        self.feature_dim = feature_extractor.state_dict()[list(feature_extractor.state_dict().keys())[-2]].shape[0]
        self.domain_to_idx = dataset.domain_to_idx
        if target_domain not in self.domain_to_idx.keys():
            raise ValueError(
                "The given target domain %s not in the dataset! The available domain names are %s"
                % (target_domain, self.domain_to_idx.keys())
            )
        self.target_domain = target_domain
        self.target_label = self.domain_to_idx[target_domain]
        self.base_params = base_params

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)

        return x

    def compute_loss(self, batch, split_name="valid"):
        raise NotImplementedError("Loss needs to be defined.")


class M3SDATrainer(BaseMultiSourceTrainer):
    """Moment matching for multi-source domain adaptation (M3SDA).

    Reference:
        Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019). Moment matching for multi-source
        domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision
        (pp. 1406-1415).
        https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        n_classes: int,
        target_domain: str,
        k_moment: int = 3,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)
        self.classifiers = dict()
        for domain_ in self.domain_to_idx.keys():
            if domain_ != target_domain:
                self.classifiers[domain_] = task_classifier(self.feature_dim, n_classes)
        # init classifiers as nn.ModuleDict, otherwise it will not be optimized
        self.classifiers = nn.ModuleDict(self.classifiers)
        self.k_moment = k_moment

    def compute_loss(self, batch, split_name="valid"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        moment_loss = self._compute_domain_dist(phi_x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)[0]
        tgt_idx = torch.where(domain_labels == self.target_label)[0]
        cls_loss, ok_src = self._compute_cls_loss(phi_x[src_idx], y[src_idx], domain_labels[src_idx])
        if len(tgt_idx) > 0:
            y_tgt_hat = _average_cls_output(phi_x[tgt_idx], self.classifiers)
            _, ok_tgt = losses.cross_entropy_logits(y_tgt_hat, y[tgt_idx])
        else:
            ok_tgt = 0.0

        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()
        task_loss = cls_loss

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": moment_loss,
        }

        return task_loss, moment_loss, log_metrics

    def _compute_cls_loss(self, x, y, domain_labels: torch.Tensor):
        if len(y) == 0:
            return 0.0, 0.0
        else:
            cls_loss = 0.0
            ok_src = []
            n_src = 0
            for domain_ in self.domain_to_idx.keys():
                if domain_ == self.target_domain:
                    continue
                domain_idx = torch.where(domain_labels == self.domain_to_idx[domain_])[0]
                cls_output = self.classifiers[domain_](x[domain_idx])
                loss_cls_, ok_src_ = losses.cross_entropy_logits(cls_output, y[domain_idx])
                cls_loss += loss_cls_
                ok_src.append(ok_src_)
                n_src += 1
            cls_loss = cls_loss / n_src
            ok_src = torch.cat(ok_src)
            return cls_loss, ok_src

    def _compute_domain_dist(self, x, domain_labels):
        """Compute the k-th order moment divergence

        Args:
            x (torch.Tensor): input data, shape (n_samples, n_features)
            domain_labels (torch.Tensor): labels indicating which domain the instance is from, shape (n_samples,)

        Returns:
            torch.Tensor: divergence
        """

        moment_loss = 0
        for i in range(self.k_moment):
            moment_loss += losses._moment_k(x, domain_labels, i + 1)

        return moment_loss


class _DINTrainer(BaseMultiSourceTrainer):
    """Domain independent network (DIN). It is under development and will be updated with references later."""

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        n_classes: int,
        target_domain: str,
        kernel: str = "linear",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)
        self.kernel = kernel
        self.n_domains = len(self.domain_to_idx.values())
        self.classifier = task_classifier(self.feature_dim, n_classes)
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def compute_loss(self, batch, split_name="valid"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        loss_dist = self._compute_domain_dist(phi_x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)[0]
        tgt_idx = torch.where(domain_labels == self.target_label)[0]
        cls_output = self.classifier(phi_x)
        loss_cls, ok_src = losses.cross_entropy_logits(cls_output[src_idx], y[src_idx])
        _, ok_tgt = losses.cross_entropy_logits(cls_output[tgt_idx], y[tgt_idx])

        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": loss_dist,
        }

        return task_loss, loss_dist, log_metrics

    def _compute_domain_dist(self, x, domain_labels):
        if self.kernel == "linear":
            kx = torch.mm(x, x.T)
        else:
            raise ValueError("Other kernels have not been implemented yet!")
        domain_label_mat = one_hot(domain_labels, num_classes=self.n_domains)
        domain_label_mat = domain_label_mat.float()
        ky = torch.mm(domain_label_mat, domain_label_mat.T)
        return losses.hsic(kx, ky, device=self.device)


class MFSANTrainer(BaseMultiSourceTrainer):
    """Multiple Feature Spaces Adaptation Network (MFSAN)

    Reference: Zhu, Y., Zhuang, F. and Wang, D., 2019, July. Aligning domain-specific distribution and classifier
        for cross-domain classification from multiple sources. In AAAI.
        https://ojs.aaai.org/index.php/AAAI/article/view/4551

    Original implementation: https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA/MFSAN
    """

    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        n_classes: int,
        target_domain: str,
        domain_feat_dim: int = 100,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        input_dimension: int = 2,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)
        self.classifiers = dict()
        self.domain_net = dict()
        self.src_domains = []
        for domain_ in dataset.domain_to_idx.keys():
            if domain_ != self.target_domain:
                self.domain_net[domain_] = _Bottleneck(
                    self.feature_dim, domain_feat_dim, input_dimension=input_dimension
                )
                self.classifiers[domain_] = task_classifier(domain_feat_dim, n_classes)
                self.src_domains.append(domain_)
        self.classifiers = nn.ModuleDict(self.classifiers)
        self.domain_net = nn.ModuleDict(self.domain_net)
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def compute_loss(self, batch, split_name="valid"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        tgt_idx = torch.where(domain_labels == self.target_label)[0]
        n_src = len(self.src_domains)
        domain_dist = 0
        loss_cls = 0
        ok_src = []
        for src_domain in self.src_domains:
            src_domain_idx = torch.where(domain_labels == self.domain_to_idx[src_domain])[0]
            phi_src = self.domain_net[src_domain].forward(phi_x[src_domain_idx])
            phi_tgt = self.domain_net[src_domain].forward(phi_x[tgt_idx])
            kernels = losses.gaussian_kernel(
                phi_src,
                phi_tgt,
                kernel_mul=self._kernel_mul,
                kernel_num=self._kernel_num,
            )
            domain_dist += losses.compute_mmd_loss(kernels, len(phi_src))
            y_src_hat = self.classifiers[src_domain](phi_src)
            loss_cls_, ok_src_ = losses.cross_entropy_logits(y_src_hat, y[src_domain_idx])
            loss_cls += loss_cls_
            ok_src.append(ok_src_)

        domain_dist += self.cls_discrepancy(phi_x[tgt_idx])
        loss_cls = loss_cls / n_src
        ok_src = torch.cat(ok_src)

        y_tgt_hat = self._get_avg_cls_output(phi_x[tgt_idx])
        _, ok_tgt = losses.cross_entropy_logits(y_tgt_hat, y[tgt_idx])

        ok_src = ok_src.double().mean()
        ok_tgt = ok_tgt.double().mean()
        task_loss = loss_cls

        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": domain_dist,
        }

        return task_loss, domain_dist, log_metrics

    def _get_cls_output(self, x):
        return [self.classifiers[key](self.domain_net[key](x)) for key in self.classifiers]

    def _get_avg_cls_output(self, x):
        cls_output = self._get_cls_output(x)

        return torch.stack(cls_output).mean(0)

    def cls_discrepancy(self, x):
        """Compute discrepancy between all classifiers' probabilistic outputs"""
        cls_output = self._get_cls_output(x)
        n_domains = len(cls_output)
        cls_disc = 0
        for i in range(n_domains - 1):
            for j in range(i + 1, n_domains):
                cls_disc_ = nn.functional.softmax(cls_output[i], dim=1) - nn.functional.softmax(cls_output[j], dim=1)
                cls_disc += torch.mean(torch.abs(cls_disc_))

        return cls_disc * 2 / (n_domains * (n_domains - 1))


class CoIRLS(BaseEstimator, ClassifierMixin):
    """Covariate-Independence Regularized Least Squares (CoIRLS)

    Args:
        kernel (str, optional): {"linear", "rbf", "poly"}. Kernel to use. Defaults to "linear".
        kernel_kwargs (dict or None, optional): Hyperparameter for the kernel. Defaults to None.
        alpha (float, optional): Hyperparameter of the l2 (Ridge) penalty. Default to 1.0.
        lambda_ (float, optional): Hyperparameter of the covariate dependence.  Default to 1.0.

    Reference:
        [1] Zhou, S., 2022. Interpretable Domain-Aware Learning for Neuroimage Classification (Doctoral dissertation,
            University of Sheffield).
        [2] Zhou, S., Li, W., Cox, C.R., & Lu, H. (2020). Side Information Dependence as a Regularizer for Analyzing
            Human Brain Conditions across Cognitive Experiments. AAAI 2020, New York, USA.
    """

    def __init__(self, kernel="linear", kernel_kwargs=None, alpha=1.0, lambda_=1.0):
        super().__init__()
        self.kernel = kernel
        self.model = None
        self.alpha = alpha
        self.lambda_ = lambda_
        if kernel_kwargs is None:
            self.kernel_kwargs = dict()
        else:
            self.kernel_kwargs = kernel_kwargs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.losses = {"ovr": [], "pred": [], "code": [], "reg": []}
        self.x = None
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        self.coef_ = None

    def fit(self, x, y, covariates):
        """fit a model with input data, labels and covariates

        Args:
            x (np.ndarray or tensor): shape (n_samples, n_features)
            y (np.ndarray or tensor): shape (n_samples, )
            covariates (np.ndarray or tensor): (n_samples, n_covariates)
        """
        self._label_binarizer.fit(y)
        x = torch.as_tensor(x)
        x = torch.cat([x, torch.ones(x.shape[0], 1)], 1)
        y = torch.as_tensor(y)
        krnl_x = torch.as_tensor(
            pairwise_kernels(x.detach().numpy(), metric=self.kernel, filter_params=True, **self.kernel_kwargs),
            dtype=torch.float,
        )
        n_samples = x.shape[0]
        n_classes = torch.unique(y).shape[0]
        n_labeled = y.shape[0]
        unit_mat = torch.eye(n_samples)
        ctr_mat = unit_mat - 1.0 / n_samples * torch.ones((n_samples, n_samples))
        mat_j = torch.zeros((n_samples, n_samples))
        mat_j[:n_labeled, :n_labeled] = torch.eye(n_labeled)

        covariates = torch.as_tensor(covariates, dtype=torch.float)
        krnl_cov = torch.mm(covariates, covariates.T)

        if n_classes == 2:
            mat_y = torch.zeros((n_samples, 1))
        else:
            mat_y = torch.zeros((n_samples, n_classes))
        mat_y[:n_labeled, :] = torch.as_tensor(self._label_binarizer.fit_transform(y))
        mat_y = torch.as_tensor(mat_y)

        mat_q = torch.mm(mat_j, krnl_x) + self.alpha * unit_mat
        mat_q += self.lambda_ * multi_dot((ctr_mat, krnl_cov, ctr_mat, krnl_x))

        self.coef_ = torch.linalg.solve(mat_q, mat_y)

        self.x = x

    def predict(self, x):
        """Predict labels for data x

        Args:
            x (np.ndarray or tensor): Samples need prediction, shape (n_samples, n_features)

        Returns:
            y (np.ndarray): Predicted labels, shape (n_samples, )
        """
        out = self.decision_function(x)
        if self._label_binarizer.y_type_ == "binary":
            pred = self._label_binarizer.inverse_transform(torch.sign(out).view(-1))
        else:
            pred = self._label_binarizer.inverse_transform(out)

        return pred

    def decision_function(self, x):
        """Compute decision scores for data x

        Args:
            x (np.ndarray or tensor): Samples need decision scores, shape (n_samples, n_features)

        Returns:
            scores (np.ndarray): Decision scores, shape (n_samples, )
        """
        x = torch.as_tensor(x)
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        krnl_x = torch.as_tensor(
            pairwise_kernels(
                x.detach().numpy(),
                self.x.detach().numpy(),
                metric=self.kernel,
                filter_params=True,
                **self.kernel_kwargs,
            ),
            dtype=torch.float,
        )

        return torch.mm(krnl_x, self.coef_)


########################################################################################################################
# Below is the Implementation of Transformer -> Maximum independence domain adaptation -> Estimator Pipeline
#
# References:
#     [1] Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace using domain features and
#         independence maximization. IEEE transactions on cybernetics, 48(1), pp.288-299.
# Developer: Shuo Zhou, shuo.zhou@sheffield.ac.uk, Lalu Muhammad Riza Rizky, l.m.rizky@sheffield.ac.uk
########################################################################################################################


INV_REG_COEF = np.logspace(start=-15, stop=15, num=30 + 1, base=2)

CLASSIFIERS = {
    "lda": LinearDiscriminantAnalysis(),
    "lr": LogisticRegression(),
    "linear_svm": LinearSVC(),
    "svm": SVC(),
    "ridge": RidgeClassifier(),
}

CLASSIFIER_PARAMS = {
    "lda": {"solver": ["svd"]},
    "lr": {"C": INV_REG_COEF},
    "linear_svm": {"C": INV_REG_COEF},
    "svm": {"C": INV_REG_COEF},
    "ridge": {"alpha": 1 / (2 * INV_REG_COEF)},
}

MIDA_PARAMS = {
    "mu": 1 / (2 * INV_REG_COEF),
    "eta": 1 / (2 * INV_REG_COEF),
    "ignore_y": [True, False],
    "augment": ["pre", "post", None],
}


class MIDATrainer(BaseSearchCV, TransformerMixin):
    """
    MIDATrainer performs supervised learning with Maximum Independence Domain Adaptation (MIDA)
    by integrating an optional transformer, a domain adapter (MIDA), and an estimator into a single
    search pipeline. It supports both grid and randomized hyperparameter search.

    The training pipeline follows this order:
        [transformer (optional)] → [MIDA] → [estimator]

    Parameters
    ----------
    estimator (sklearn.base.BaseEstimator): The base estimator to train.
    param_grid (dict or list of dict): Dictionary with parameter names (str) as keys and lists of parameter settings
        to try as values. Use 'transformer__' and 'domain_adapter__' prefixes for transformer and MIDA parameters,
        respectively.
    use_mida (bool): Whether to apply MIDA for domain adaptation.
    transformer (sklearn.base.BaseEstimator, optional): Preprocessing transformer (e.g., StandardScaler or PCA).
        Must implement `fit` and `transform`.
    search_strategy (str, optional): Either "grid" or "random". Determines whether to perform grid search or randomized search.
    num_iter (int, optional): Number of sampled parameter settings for randomized search. Ignored if `search_strategy="grid"`. Default is 10.
    scoring (str, optional, callable, or dict): Scoring metric(s) to evaluate the predictions on the test set. Default is None.
    n_jobs (int, optional): Number of jobs to run in parallel. Default is None.
    pre_dispatch (int or str, optional): Controls the number of jobs that get dispatched during parallel execution.
        Default is "2*n_jobs".
    refit (bool, optional): Whether to refit the best estimator using the entire dataset. Default is True.
    cv (int, optional): Cross-validation generator, or iterable Determines the cross-validation splitting strategy. Default is None.
    verbose (int, optional): Controls the verbosity of the output. Default is 0.
    random_state (int, optional): Random seed for reproducibility. RandomState instance or None, default is None.
    error_score ('raise' or float, optional): Value to assign if an error occurs in estimator fitting. Default is np.nan.
    return_train_score (bool, optional): Whether to include training scores in the cv_results_. Default is False.

    Example
    -------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from kale.pipeline.mida_trainer import MIDATrainer
    >>> import numpy as np
    >>> X, y = make_classification(n_samples=1000, n_features=20)
    >>> group_labels = np.random.randint(0, 2, size=(1000, 1))  # Domain labels
    >>> estimator = RandomForestClassifier()
    >>> param_grid = {
    ...     "estimator__n_estimators": [50, 100],
    ...     "domain_adapter__num_components": [2, 5],
    ...     "transformer__with_mean": [True, False],
    ... }
    >>> trainer = MIDATrainer(estimator, param_grid, transformer=StandardScaler(), cv=5)
    >>> trainer.fit(X,y,group_labels=group_labels)
    >>> predictions = trainer.predict(X)
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
        self._group_label_encoder = None

    def adapt(self, x, group_labels=None):
        """Adapt the estimator to the given data.

        Args:
            x (array-like): The input data.
            group_labels (array-like): Categorical variables representing domain or grouping factors with shape
                (num_samples, num_factors). Preprocessing (e.g., one-hot encoding) must be applied in advance.
        Returns:
            array-like: The transformed or adapted data.
        """
        check_is_fitted(self)
        if group_labels is not None and self._group_label_encoder is not None:
            if (1 in group_labels.shape and group_labels.ndim == 2) or (group_labels.ndim == 1):
                group_labels = self._group_label_encoder.transform(group_labels.reshape(-1, 1))
        if self.transformer is not None:
            x = self.best_transformer_.transform(x)

        if self.use_mida:
            x = self.best_mida_.transform(x, group_labels)

        return x

    def score(self, x, y=None, group_labels=None, **params):
        """Compute the score of the estimator on the given data.

        Args:
            x (array-like): The data matrix to be socred, shape (num_samples, num_features).
            y (array-like): The target values.
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
            **params: Additional parameters for the estimator.
        Returns:
            float: The score of the estimator.
        """
        check_is_fitted(self)
        x = self.adapt(x, group_labels)
        return super().score(x, y, **params)

    def score_samples(self, x, group_labels=None):
        """Compute the log-likelihood of the samples.

        Args:
            x (array-like): The data matrix to be socred, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors with shape
                (num_samples, num_factors). Preprocessing (e.g., one-hot encode domain, gender, or age groups)
                must be applied in advance.
        Returns:
            array-like: The log-likelihood of the samples.
        """
        x = self.adapt(x, group_labels)
        return super().score_samples(x)

    def predict(self, x, group_labels=None):
        """Predict using the best pipeline.

        Args:
            x (array-like): The data matrix for which we want to get the predictions, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
        Returns:
            array-like: The predicted target.
        """
        x = self.adapt(x, group_labels)
        return super().predict(x)

    def predict_proba(self, x, group_labels=None):
        """Predict class probabilities using the best pipeline.

        Args:
            x (array-like): The data matrix for which we want to get the probability of the predictions,
                shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
        Returns:
            array-like: The predicted class probabilities.
        """
        x = self.adapt(x, group_labels)
        return super().predict_proba(x)

    def predict_log_proba(self, x, group_labels=None):
        """Predict log class probabilities using the best pipeline.

        Args:
            x (array-like): The data matrix for which we want to get the log probability of the predictions,
                shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
        Returns:
            array-like: The predicted log class probabilities.
        """
        x = self.adapt(x, group_labels)
        return super().predict_log_proba(x)

    def decision_function(self, x, group_labels=None):
        """Compute the decision function using the best pipeline."""
        x = self.adapt(x, group_labels)
        return super().decision_function(x)

    def transform(self, x, group_labels=None):
        """Transform the data using the best pipeline, equivalent to self.best_estimator_.inverse_transform(x)."""
        x = self.adapt(x, group_labels)
        return super().transform(x)

    def inverse_transform(self, x):
        """Inverse transform the data using the best pipeline.
        Note that this method is not compatible for MIDA when `augment=True`.

        Args:
            x (array-like): The data matrix to be reverted.
        Returns:
            array-like: The inverse transformed data.
        """
        check_is_fitted(self)
        x = super().inverse_transform(x)  # equivalent to self.best_estimator_.inverse_transform(x)

        if hasattr(self.best_mida_, "inverse_transform"):
            x = self.best_mida_.inverse_transform(x)

        if self.transformer is not None and hasattr(self.best_transformer_, "inverse_transform"):
            x = self.best_transformer_.inverse_transform(x)

        return x

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""

        # Can't be replaced since it is used to validate the input data
        # by scikit-learn, default across all estimators

        # Trick to call the try exception block of the parent class
        super().n_features_in_
        if self.transformer is not None:
            return self.best_transformer_.n_features_in_

        if self.use_mida:
            return self.best_mida_.n_features_in_

        # Call again to return the original value
        # Still need to call on the first part to do validation
        return super().n_features_in_

    @property
    def groups_(self):
        return self._group_label_encoder.classes_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y=None, group_labels=None, **params):
        """Fit and tune the pipeline with the given data.

        Args:
            x (array-like): The input data.
            y (array-like): The target values.
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
            **params: Additional parameters for the estimator.
        Returns:
            self: The fitted trainer.
        """

        estimator = self.estimator
        transformer = self.transformer
        mida = MIDA() if self.use_mida else None
        scorers, refit_metric = self._get_scorers()
        if (1 in group_labels.shape and group_labels.ndim == 2) or (group_labels.ndim == 1):
            params["groups"] = group_labels.copy()
            self._group_label_encoder = LabelBinarizer(sparse_output=False)
            group_labels = self._group_label_encoder.fit_transform(group_labels.reshape(-1, 1))

        x, y, group_labels = indexable(x, y, group_labels)
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
                        transformer=clone(base_transformer) if base_transformer else None,
                        domain_adapter=clone(base_mida) if base_mida else None,
                        group_labels=group_labels,
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
                x = self.best_mida_.fit_transform(x, y, group_labels)
            elif self.use_mida:
                x = self.best_mida_.fit_transform(x, group_labels=group_labels)

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


class AutoMIDAClassificationTrainer(MetaEstimatorMixin, BaseEstimator):
    """
    AutoMIDAClassificationTrainer builds a classification pipeline using MIDA (Maximum Independence Domain Adaptation)
    and a classifier, with automated estimator selection if not specified.

    This meta-estimator wraps the MIDATrainer, simplifying usage for classification tasks while supporting
    domain adaptation and optional transformation steps.

    Args:
        classifier (str, optional): Classifier type or "auto" to automatically select one.
            Supported strings include {"lr", "linear_svm", "svm", "ridge"}. Default is "auto".
        use_mida (bool, optional): Whether to use MIDA for domain adaptation. Default is True.
        nonlinear (bool, optional): Whether to enable nonlinear MIDA. Default is False.
        transformer (sklearn.base.BaseEstimator or None, optional): Optional transformer to apply before MIDA.
            Must implement `fit` and `transform`. Default is None.
        search_strategy (str, optional): "grid" or "random" search for hyperparameter optimization. Default is "random".
        num_search_iter (int, optional): Number of parameter settings to sample in random search. Default is 10.
        num_solver_iter (int, optional): Maximum iterations for the solver. Default is 100.
        scoring (str or callable or list or tuple or dict, optional): Scoring metric(s) to use. Default is None.
        n_jobs (int or None, optional): Number of parallel jobs to run. Default is None.
        pre_dispatch (int or str, optional): Controls the number of jobs dispatched during parallel execution.
            Default is "2*n_jobs".
        refit (bool, optional): Whether to refit the best estimator. Default is True.
        cv (int or cross-validation generator, optional): Cross-validation splitting strategy. Default is None.
        verbose (int, optional): Verbosity level. Default is 0.
        random_state (int or np.random.RandomState, optional): Seed or RNG for reproducibility. Default is None.
        error_score ('raise' or numeric, optional): Value to assign to the score if an error occurs in estimator fitting.
            Default is np.nan.
        return_train_score (bool, optional): Whether to include training scores in the results. Default is False.
    """

    _parameter_constraints = {
        "classifier": [StrOptions({"auto"} | set(CLASSIFIERS))],
        "use_mida": ["boolean"],
        "nonlinear": ["boolean"],
        "transformer": [HasMethods(["fit", "transform"]), None],
        "search_strategy": [StrOptions({"grid", "random"})],
        "num_search_iter": [Interval(Integral, 1, None, closed="left")],
        "num_solver_iter": [Interval(Integral, 1, None, closed="left")],
        "scoring": MIDATrainer._parameter_constraints["scoring"],
        "n_jobs": MIDATrainer._parameter_constraints["n_jobs"],
        "pre_dispatch": MIDATrainer._parameter_constraints["pre_dispatch"],
        "refit": MIDATrainer._parameter_constraints["refit"],
        "cv": MIDATrainer._parameter_constraints["cv"],
        "verbose": MIDATrainer._parameter_constraints["verbose"],
        "random_state": MIDATrainer._parameter_constraints["random_state"],
        "error_score": MIDATrainer._parameter_constraints["error_score"],
        "return_train_score": MIDATrainer._parameter_constraints["return_train_score"],
    }

    def __init__(
        self,
        classifier="auto",
        use_mida=True,
        nonlinear=False,
        transformer=None,
        search_strategy="random",
        num_search_iter=10,
        num_solver_iter=100,
        scoring=None,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        refit=True,
        cv=None,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.classifier = classifier
        self.use_mida = use_mida
        self.nonlinear = nonlinear
        self.transformer = transformer
        self.search_strategy = search_strategy
        self.num_search_iter = num_search_iter
        self.num_solver_iter = num_solver_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.group_labels_encoder = None

    @property
    def coef_(self):
        """Original coefficients of the best estimator in original feature space.
        Return original feature space coefficients with shape (n_classes, n_features).
        If MIDA is used, the coefficients are transformed to the original feature space.
        Only available when `nonlinear=False`.
        """
        check_is_fitted(self)

        if self.nonlinear:
            raise ValueError("coef_ is not available when `nonlinear=True`.")

        augment = None
        coef = self.best_classifier_.coef_

        if self.use_mida:
            augment = self.best_mida_.augment
            mida_coef = self.best_mida_.orig_coef_

        if self.use_mida and augment == "post":
            # Split the coefficients for MIDA features and factors
            n_out = self.best_mida_._n_features_out
            mida_clf_coef, factor_coef = np.split(coef, [n_out], axis=1)

            # Dot product the MIDA coefficients with the original coefficients
            # then concatenate with the factor coefficients
            coef = np.concatenate((mida_clf_coef @ mida_coef, factor_coef), axis=1)
        elif self.use_mida:
            # Dot product the MIDA coefficients with the original coefficients
            # then concatenate with the factor coefficients
            coef = coef @ mida_coef

        return coef

    @property
    def cv_results_(self):
        """Results from cross-validation. A dict with keys as column headers and values as sequences."""
        check_is_fitted(self)
        return self.trainer_.cv_results_

    @property
    def best_classifier_(self):
        """Best estimator found by the search."""
        check_is_fitted(self)

        if self.classifier == "auto":
            # If the classifier is auto, we need to extract the classifier from the pipeline
            return self.trainer_.best_estimator_["classifier"]

        return self.trainer_.best_estimator_

    @property
    def best_transformer_(self):
        """Best transformer found by the search, if any."""
        check_is_fitted(self)
        return self.trainer_.best_transformer_

    @property
    def best_mida_(self):
        """Best MIDA component found by the search, if used."""
        check_is_fitted(self)
        return self.trainer_.best_mida_

    @property
    def best_score_(self):
        """Mean cross-validated score of the best_estimator."""
        check_is_fitted(self)
        return self.trainer_.best_score_

    @property
    def best_params_(self):
        """Parameter setting that gave the best results on the hold out data."""
        check_is_fitted(self)
        return self.trainer_.best_params_

    @property
    def best_index_(self):
        """Index (of `cv_results_`) which corresponds to the best candidate parameter setting."""
        check_is_fitted(self)
        return self.trainer_.best_index_

    @property
    def scorer_(self):
        """Scorer function or dict used to evaluate the predictions on the test set."""
        check_is_fitted(self)
        return self.trainer_.scorer_

    @property
    def n_splits_(self):
        """Number of cross-validation splits (folds)."""
        check_is_fitted(self)
        return self.trainer_.n_splits_

    @property
    def refit_time_(self):
        """Time (in seconds) for refitting the best estimator on the whole dataset."""
        check_is_fitted(self)
        return self.trainer_.refit_time_

    @property
    def multimetric_(self):
        """Whether multiple metrics are used for scoring."""
        check_is_fitted(self)
        return self.trainer_.multimetric_

    @property
    def classes_(self):
        """Classes seen during `fit`."""
        check_is_fitted(self)
        return self.trainer_.classes_

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        check_is_fitted(self)
        return self.trainer_.n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during `fit`."""
        check_is_fitted(self)
        return self.trainer_.feature_names_in_

    @property
    def groups_(self):
        return self.trainer_.groups_

    def _get_classifier_and_grid(self):
        """Get the classifier and parameter grid for the selected classifier.

        Returns:
            tuple:
                - classifier: The classifier instance.
                - param_grid: The parameter grid for the classifier.
        """
        base_classifier = clone(CLASSIFIERS.get(self.classifier, LogisticRegression()))
        if hasattr(base_classifier, "max_iter"):
            base_classifier.set_params(max_iter=self.num_solver_iter, random_state=self.random_state)

        if self.classifier != "auto":
            return base_classifier, CLASSIFIER_PARAMS[self.classifier]

        # Workaround to allow classifier selection
        base_classifier = Pipeline([("classifier", base_classifier)])
        param_grid = []

        for name, classifier in CLASSIFIERS.items():
            classifier = clone(classifier)
            if hasattr(classifier, "max_iter"):
                classifier.set_params(max_iter=self.num_solver_iter)
            grid = {"classifier": [classifier]}
            grid.update({f"classifier__{param}": value for param, value in CLASSIFIER_PARAMS[name].items()})

            if self.nonlinear and name == "svm":
                grid["classifier__kernel"] = ["linear", "rbf"]

            param_grid.append(grid)

        return base_classifier, param_grid

    def _get_mida_and_grid(self):
        """Get the MIDA component and parameter grid for MIDA.

        Returns:
            dict: Parameter grid for MIDA.
        """
        if not self.use_mida:
            return {}

        param_grid = {f"domain_adapter__{param}": value for param, value in MIDA_PARAMS.items()}

        if self.nonlinear:
            param_grid["domain_adapter__kernel"] = ["linear", "rbf"]

        return param_grid

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y=None, group_labels=None, **params):
        """
        Fit the AutoMIDAClassificationTrainer pipeline to the data.

        Internally configures MIDATrainer with selected estimator, transformer, and domain adapter.

        Args:
            x (array-like): Input data matrix, shape (num_samples, num_features).
            y (array-like): Target labels, shape (num_samples,).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
            **params: Additional parameters forwarded to the MIDATrainer.

        Returns:
            self: Fitted trainer instance.
        """

        classifier, classifier_grid = self._get_classifier_and_grid()
        mida_grid = self._get_mida_and_grid()

        # Aggregate parameter grids
        if self.classifier == "auto":
            param_grid = [{**grid, **mida_grid} for grid in classifier_grid]
        else:
            param_grid = {**classifier_grid, **mida_grid}

        # Initialize MIDATrainer with the selected classifier and parameter grid
        trainer = MIDATrainer(
            estimator=classifier,
            param_grid=param_grid,
            use_mida=self.use_mida,
            transformer=self.transformer,
            search_strategy=self.search_strategy,
            num_iter=self.num_search_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            refit=self.refit,
            cv=self.cv,
            verbose=self.verbose,
            random_state=self.random_state,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        trainer.fit(x, y, group_labels=group_labels, **params)

        self.trainer_ = trainer

        return self

    def adapt(self, x, group_labels=None):
        """
        Adapt the data using the trained pipeline.

        This applies the fitted transformer (if provided) and the fitted MIDA component
        to the input features, producing domain-adapted representations.

        Args:
            x (array-like): Input data to transform, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Adapted feature matrix.
        """
        check_is_fitted(self)
        return self.trainer_.adapt(x, group_labels=group_labels)

    def predict(self, x, group_labels=None):
        """
        Predict class labels using the trained pipeline.

        This applies domain adaptation (transformer + MIDA) before calling the classifier.

        Args:
            x (array-like): The data matrix for which we want to get the predictions, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Predicted class labels.
        """
        check_is_fitted(self)
        return self.trainer_.predict(x, group_labels=group_labels)

    def predict_proba(self, x, group_labels=None):
        """
        Predict class probabilities using the trained pipeline.

        This applies domain adaptation before computing probabilities from the classifier.

        Args:
            x (array-like): The data matrix for which we want to get the probability of the predictions,
                shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Class probability estimates.
        """
        check_is_fitted(self)
        return self.trainer_.predict_proba(x, group_labels=group_labels)

    def predict_log_proba(self, x, group_labels=None):
        """
        Predict log class probabilities using the trained pipeline.

        This applies domain adaptation before computing log probabilities.

        Args:
            x (array-like): The data matrix for which we want to get the log probability of the predictions,
                shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Log probability estimates.
        """
        check_is_fitted(self)
        return self.trainer_.predict_log_proba(x, group_labels=group_labels)

    def decision_function(self, x, group_labels=None):
        """
        Compute the decision function using the trained pipeline.

        This applies domain adaptation before computing decision values from the classifier.

        Args:
            x (array-like): The data matrix for which we want to get the decision scores, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Confidence scores for each input.
        """
        check_is_fitted(self)
        return self.trainer_.decision_function(x, group_labels=group_labels)

    def score(self, x, y=None, group_labels=None, **params):
        """
        Compute the score of the trained pipeline on given data.

        This applies domain adaptation before scoring with the estimator.

        Args:
            x (array-like): The data matrix to be socred, shape (num_samples, num_features).
            y (array-like): True labels, shape (num_samples,).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).
            **params: Additional scoring parameters.

        Returns:
            float: Score value.
        """
        check_is_fitted(self)
        return self.trainer_.score(x, y=y, group_labels=group_labels, **params)

    def score_samples(self, x, group_labels=None):
        """
        Compute the log-likelihood of samples using the trained pipeline.

        This applies domain adaptation before estimating sample likelihoods.

        Args:
            x (array-like): The data matrix to be socred, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Log-likelihood scores.
        """
        check_is_fitted(self)
        return self.trainer_.score_samples(x, group_labels=group_labels)

    def transform(self, x, group_labels=None):
        """
        Transform data using the trained pipeline.

        Applies transformer and MIDA components in sequence.

        Args:
            x (array-like): The data matrix to be transformed, shape (num_samples, num_features).
            group_labels (array-like): Categorical variables representing domain or grouping factors, shape (num_samples,).

        Returns:
            array-like: Transformed features.
        """
        check_is_fitted(self)
        return self.trainer_.transform(x, group_labels=group_labels)

    def inverse_transform(self, x):
        """
        Inverse transform the data using the trained pipeline.

        Applies inverse transformations from MIDA and transformer if supported.

        Args:
            x (array-like): The data matrix to be reverted.

        Returns:
            array-like: Reconstructed input features.
        """
        check_is_fitted(self)
        return self.trainer_.inverse_transform(x)

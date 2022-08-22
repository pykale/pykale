# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk/sz144@outlook.com
# =============================================================================
"""Multi-source domain adaptation pipelines
"""

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from torch.linalg import multi_dot
from torch.nn.functional import one_hot

import kale.predict.losses as losses
from kale.embed.image_cnn import _Bottleneck
from kale.pipeline.domain_adapter import BaseAdaptTrainer, get_aggregated_metrics


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
        self, dataset, feature_extractor, task_classifier, n_classes: int, target_domain: str, **base_params,
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

    def validation_epoch_end(self, outputs):
        metrics_to_log = (
            "valid_loss",
            "valid_source_acc",
            "valid_target_acc",
            "valid_domain_acc",
        )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        metrics_at_test = (
            "test_loss",
            "test_source_acc",
            "test_target_acc",
            "test_domain_acc",
        )
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)


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
                phi_src, phi_tgt, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,
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
        """Compute discrepancy between all classifiers' probabilistic outputs
        """
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
        alpha (float, optional): Hyperparameter of the l2 (Ridge) penalty. Defaults to 1.0.
        lambda_ (float, optional): Hyperparameter of the covariate dependence.  Defaults to 1.0.

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

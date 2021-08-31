# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import torch
import torch.nn as nn
from torch.nn.functional import one_hot

import kale.predict.losses as losses
from kale.pipeline.domain_adapter import BaseAdaptTrainer, get_aggregated_metrics


def create_ms_adapt_trainer(method: str, dataset, feature_extractor, task_classifier, **train_params):
    method_dict = {"M3SDA": M3SDATrainer, "DIN": _DINTrainer, "MFSAN": MFSANTrainer}
    method = method.upper()
    if method not in method_dict.keys():
        raise ValueError("Unsupported multi-source domain adaptation methods %s" % method)
    else:
        return method_dict[method](dataset, feature_extractor, task_classifier, **train_params)


def _moment_k(x: torch.Tensor, domain_labels: torch.Tensor, k_order=2):
    """Compute k-th moment distance

    Args:
        x (torch.Tensor): input data, shape (n_samples, n_features)
        domain_labels (torch.Tensor): labels indicate the instance from which domain, shape (n_samples,)
        k_order (int, optional): moment order. Defaults to 2.

    Returns:
        torch.Tensor: k-th moment distance
    """
    unique_domain_ = torch.unique(domain_labels)
    n_unique_domain_ = len(unique_domain_)
    x_k_order = []
    for domain_label_ in unique_domain_:
        domain_idx = torch.where(domain_labels == domain_label_)
        if k_order == 1:
            x_k_order.append(x[domain_idx].mean(0))
        else:
            x_k_order.append(((x[domain_idx] - x[domain_idx].mean(0)) ** k_order).mean(0))
    moment_sum = 0
    n_pair = 0
    for i in range(n_unique_domain_):
        for j in range(i + 1, n_unique_domain_):
            moment_sum += losses.euclidean(x_k_order[i], x_k_order[j])
            n_pair += 1
    return moment_sum / n_pair


def _average_cls_output(x, classifiers: nn.ModuleDict):
    cls_output = [classifiers[key](x) for key in classifiers]

    return torch.stack(cls_output).mean(0)


class BaseMultiSourceTrainer(BaseAdaptTrainer):
    def __init__(
        self, dataset, feature_extractor, task_classifier, n_classes: int, target_domain: str, **base_params,
    ):
        """Base class for all domain adaptation architectures

        Args:
            dataset (kale.loaddata.multi_domain): the multi-domain datasets to be used for train, validation, and tests.
            feature_extractor (torch.nn.Module): the feature extractor network
            task_classifier (torch.nn.Module): the task classifier network
            n_classes (int): number of classes
            target_domain (str): target domain name
        """
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)
        self.n_classes = n_classes
        self.feature_dim = feature_extractor.state_dict()[list(feature_extractor.state_dict().keys())[-2]].shape[0]
        self.domain_to_idx = dataset.domain_to_idx
        if target_domain not in self.domain_to_idx.keys():
            raise ValueError(
                "The given target domain %s not in the dataset! The available domain names are %s"
                % self.domain_to_idx.keys()
            )
        self.target_domain = target_domain
        self.target_label = self.domain_to_idx[target_domain]
        self.base_params = base_params

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        # x = x.view(x.size(0), -1)
        return x

    def compute_loss(self, batch, split_name="V"):
        raise NotImplementedError("Loss needs to be defined.")

    def validation_epoch_end(self, outputs):
        metrics_to_log = (
            "val_loss",
            "V_source_acc",
            "V_target_acc",
            "V_domain_acc",
        )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        metrics_at_test = (
            "test_loss",
            "Te_source_acc",
            "Te_target_acc",
            "Te_domain_acc",
        )
        log_dict = get_aggregated_metrics(metrics_at_test, outputs)

        for key in log_dict:
            self.log(key, log_dict[key], prog_bar=True)


class M3SDATrainer(BaseMultiSourceTrainer):
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
        """Moment matching for multi-source domain adaptation.

        Reference:
            Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019). Moment matching for multi-source
            domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision
            (pp. 1406-1415).
        """
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)
        self.classifiers = dict()
        for domain_ in self.domain_to_idx.keys():
            if domain_ != target_domain:
                self.classifiers[domain_] = task_classifier(self.feature_dim, n_classes)
        # init classifiers as nn.ModuleDict, otherwise it will not be optimized
        self.classifiers = nn.ModuleDict(self.classifiers)
        self.k_moment = k_moment

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        moment_loss = self._compute_domain_dist(phi_x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tgt_idx = torch.where(domain_labels == self.target_label)
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
                domain_idx = torch.where(domain_labels == self.domain_to_idx[domain_])
                cls_output = self.classifiers[domain_](x[domain_idx])
                loss_cls_, ok_src_ = losses.cross_entropy_logits(cls_output, y[domain_idx])
                cls_loss += loss_cls_
                ok_src.append(ok_src_)
                n_src += 1
            cls_loss = cls_loss / n_src
            ok_src = torch.cat(ok_src)
            return cls_loss, ok_src

    def _compute_domain_dist(self, x, domain_labels):
        """Compute k-th order moment divergence

        Args:
            x (torch.Tensor): input data, shape (n_samples, n_features)
            domain_labels (torch.Tensor): labels indicate the instance from which domain, shape (n_samples,)

        Returns:
            torch.Tensor: divergence
        """

        moment_loss = 0
        for i in range(self.k_moment):
            moment_loss += _moment_k(x, domain_labels, i + 1)

        return moment_loss


class _DINTrainer(BaseMultiSourceTrainer):
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
        """Domain independent network.

        """
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)
        self.kernel = kernel
        self.n_domains = len(self.domain_to_idx.values())
        self.classifier = task_classifier(self.feature_dim, n_classes)
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        loss_dist = self._compute_domain_dist(phi_x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tgt_idx = torch.where(domain_labels == self.target_label)
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
        **base_params,
    ):
        """
        Reference: Zhu, Y., Zhuang, F. and Wang, D., 2019, July. Aligning domain-specific distribution and classifier
            for cross-domain classification from multiple sources. In Proceedings of the AAAI Conference on Artificial
            Intelligence (Vol. 33, No. 01, pp. 5989-5996).

        Original implementation: https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA/MFSAN
        """
        super().__init__(dataset, feature_extractor, task_classifier, n_classes, target_domain, **base_params)

        self.classifiers = dict()
        self.sonnet = dict()
        self.src_domains = []
        for domain_ in dataset.domain_to_idx.keys():
            if domain_ != self.target_domain:
                self.sonnet[domain_] = _ADDneck(self.feature_dim, domain_feat_dim)
                self.classifiers[domain_] = task_classifier(domain_feat_dim, n_classes)
                self.src_domains.append(domain_)
        self.classifiers = nn.ModuleDict(self.classifiers)
        self.sonnet = nn.ModuleDict(self.sonnet)
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        tgt_idx = torch.where(domain_labels == self.target_label)
        n_src = len(self.src_domains)
        mmd_dist = 0
        loss_cls = 0
        ok_src = []
        for src_domain in self.src_domains:
            src_domain_idx = torch.where(domain_labels == self.domain_to_idx[src_domain])
            phi_src = self.sonnet[src_domain].forward(phi_x[src_domain_idx])
            phi_tgt = self.sonnet[src_domain].forward(phi_x[tgt_idx])
            kernels = losses.gaussian_kernel(
                phi_src, phi_tgt, kernel_mul=self._kernel_mul, kernel_num=self._kernel_num,
            )
            mmd_dist += losses.compute_mmd_loss(kernels, len(phi_src))
            y_src_hat = self.classifiers[src_domain](phi_src)
            loss_cls_, ok_src_ = losses.cross_entropy_logits(y_src_hat, y[src_domain_idx])
            loss_cls += loss_cls_
            ok_src.append(ok_src_)

        loss_cls = loss_cls / n_src
        ok_src = torch.cat(ok_src)

        y_tgt_hat = self._get_avg_cls_output(phi_x[tgt_idx])
        _, ok_tgt = losses.cross_entropy_logits(y_tgt_hat, y[tgt_idx])

        task_loss = loss_cls
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": mmd_dist,
        }

        return task_loss, mmd_dist, log_metrics

    def _get_avg_cls_output(self, x):
        cls_output = [self.classifiers[key](self.sonnet[key](x)) for key in self.classifiers]

        return torch.stack(cls_output).mean(0)


class _ADDneck(nn.Module):
    """Simple network for domain specific embedding
    Original implementation see:
     https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN/MFSAN_2src/resnet.py or
     https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN/MFSAN_3src/resnet.py
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import kale.predict.losses as losses

# from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.pipeline.domain_adapter import BaseAdaptTrainer, get_aggregated_metrics


def create_ms_adapt_trainer(method: str, dataset, feature_extractor, task_classifier, **train_params):
    method_dict = {"M3SDA": M3SDATrainer, "DIN": DINTrainer, "M": MFSANTrainer}
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


def average_cls_output(x, classifiers: Dict[int, Any]):
    cls_output = [classifiers[key](x) for key in classifiers]
    cls_output = torch.stack(cls_output)
    return cls_output.mean(0)


class BaseMultiSourceTrainer(BaseAdaptTrainer):
    def __init__(
        self,
        dataset,
        feature_extractor,
        task_classifier,
        target_label: int,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, task_classifier, **base_params)
        self.domain_to_idx = dataset.domain_to_idx
        if target_label not in self.domain_to_idx.values():
            raise ValueError("The given target label %s not in the given dataset! The available domain labels are %s"
                             % self.domain_to_idx.values())
        self.target_label = target_label

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        return x

    def compute_loss(self, batch, split_name="V"):
        raise NotImplementedError("Loss needs to be defined.")

    def validation_epoch_end(self, outputs):
        metrics_to_log = (
            "val_loss",
            "V_source_acc",
            "V_target_acc",
            "V_domain_dist",
        )
        return self._validation_epoch_end(outputs, metrics_to_log)

    def test_epoch_end(self, outputs):
        metrics_at_test = (
            "test_loss",
            "Te_source_acc",
            "Te_target_acc",
            "Te_domain_dist",
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
        target_label: int,
        k_moment: int = 5,
        **base_params,
    ):
        """Moment matching for multi-source domain adaptation. 

        Reference:    
            Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019).
            Moment matching for multi-source domain adaptation. In Proceedings of the
            IEEE/CVF International Conference on Computer Vision (pp. 1406-1415).
        """
        super().__init__(dataset, feature_extractor, task_classifier, target_label, **base_params)

        self.classifiers = dict()
        for domain_label_ in self.domain_to_idx.values():
            if domain_label_ != target_label:
                self.classifiers[domain_label_] = task_classifier
        self.k_moment = k_moment

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        moment_loss = self._compute_domain_dist(phi_x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tgt_idx = torch.where(domain_labels == self.target_label)
        cls_loss, ok_src = self._compute_cls_loss(phi_x[src_idx], y[src_idx], domain_labels[src_idx])
        if len(tgt_idx) > 0:
            y_tgt_hat = average_cls_output(phi_x[tgt_idx], self.classifiers)
            _, ok_tgt = losses.cross_entropy_logits(y_tgt_hat, y[tgt_idx])
        else:
            ok_tgt = 0.0

        task_loss = cls_loss
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_dist": moment_loss,
        }

        return task_loss, moment_loss, log_metrics

    def _compute_cls_loss(self, x, y, domain_labels: torch.Tensor):
        if len(y) == 0:
            return 0.0, 0.0
        else:
            unique_domain_ = torch.unique(domain_labels).squeeze().tolist()
            cls_loss = 0.0
            ok_src = []
            n_src = len(unique_domain_)
            for domain_label_ in unique_domain_:
                if domain_label_ == self.target_label:
                    continue
                domain_idx = torch.where(domain_labels == domain_label_)
                cls_output = self.classifiers[domain_label_](x[domain_idx])
                loss_cls_, ok_src_ = losses.cross_entropy_logits(cls_output, y[domain_idx])
                cls_loss += loss_cls_
                ok_src.append(ok_src_)
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

        # moment_loss = _moment_k(x, domain_label, 1)
        moment_loss = 0
        # print(reg_info)
        for i in range(self.k_moment):
            moment_loss += _moment_k(x, domain_labels, i + 1)

        return moment_loss


class DINTrainer(BaseMultiSourceTrainer):
    def __init__(
        self, dataset, feature_extractor, task_classifier, target_label: int, kernel: str = "linear",
        kernel_mul: float = 2.0, kernel_num: int = 5, **base_params,
    ):
        """Domain independent network. 

        """
        super().__init__(dataset, feature_extractor, task_classifier, target_label, **base_params)
        self.kernel = kernel
        self.n_domains = len(self.domain_to_idx.values())
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
            f"{split_name}_domain_dist": loss_dist,
        }

        return task_loss, loss_dist, log_metrics

    def _compute_domain_dist(self, x, domain_labels):
        if self.kernel == "linear":
            kx = torch.mm(x, x.T)
        else:
            raise ValueError("Other kernels have not been implemented yet!")
        domain_label_mat = one_hot(domain_labels, num_classes=self.n_domains)
        ky = torch.mm(domain_label_mat, domain_label_mat.T)
        return losses.hsic(kx, ky)


class MFSANTrainer(BaseMultiSourceTrainer):
    def __init__(
        self, dataset, feature_extractor, task_classifier, target_label: int, n_classes: int,
        domain_feat_dim: int = 100, kernel_mul: float = 2.0, kernel_num: int = 5, **base_params,
    ):
        """

        Reference:

        """
        super().__init__(dataset, feature_extractor, task_classifier, target_label, **base_params)

        self.classifiers = dict()
        self.sonnet = dict()
        n_input_feat = self.feat.output_size()
        for domain_label_ in dataset.domain_to_idx.values():
            if domain_label_ != target_label:
                self.classifiers[domain_label_] = task_classifier(domain_feat_dim, n_classes)
                self.sonnet[domain_label_] = ADDneck(n_input_feat, domain_feat_dim)
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        phi_x = self.forward(x)
        src_idx = torch.where(domain_labels != self.target_label)
        tgt_idx = torch.where(domain_labels == self.target_label)
        unique_src_domains = torch.unique(domain_labels[src_idx]).squeeze().tolist()
        n_src = len(unique_src_domains)
        mmd_dist = 0
        loss_cls = 0
        ok_src = []
        for src_domain in unique_src_domains:
            src_domain_idx = torch.where(domain_labels == src_domain)
            phi_src = self.sonnet[src_domain].forward(phi_x[src_domain_idx])
            phi_tgt = self.sonnet[src_domain].forward(phi_x[tgt_idx])
            kernels = losses.gaussian_kernel(phi_src, phi_tgt, kernel_mul=self._kernel_mul,
                                             kernel_num=self._kernel_num, )
            mmd_dist += losses.compute_mmd_loss(kernels, len(phi_src))
            y_src_hat = self.classifiers[src_domain](phi_src)
            loss_cls_, ok_src_ = losses.cross_entropy_logits(y_src_hat, y[src_domain_idx])
            loss_cls += loss_cls_
            ok_src.append(ok_src_)

        loss_cls = loss_cls / n_src
        ok_src = torch.cat(ok_src)

        y_tgt_hat = average_cls_output(phi_x[tgt_idx], self.classifiers)
        _, ok_tgt = losses.cross_entropy_logits(y_tgt_hat, y[tgt_idx])

        task_loss = loss_cls
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_dist": mmd_dist,
        }

        return task_loss, mmd_dist, log_metrics


class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

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

        return out

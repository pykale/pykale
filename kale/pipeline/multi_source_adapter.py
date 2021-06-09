# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

from typing import Any, Dict

import torch

import kale.predict.losses as losses

# from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.pipeline.domain_adapter import BaseAdaptTrainer, get_aggregated_metrics


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
    x_order_k = []
    for domain_label_ in unique_domain_:
        domain_idx = torch.where(domain_labels == domain_label_)
        x_order_k.append((x[domain_idx] ** k_order).mean(0))
    moment_sum = 0
    n_pair = 0
    for i in range(n_unique_domain_):
        for j in range(i + 1, n_unique_domain_):
            moment_sum += losses.euclidean(x_order_k[i], x_order_k[j])
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
        if target_label not in dataset.domain_to_idx.values():
            raise ValueError("The given target label %s not in the given dataset! The available domain labels are %s"
                             % dataset.domain_to_idx.values())
        self.target_label = target_label

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        return x

    def compute_loss(self, batch, split_name="V"):
        raise NotImplementedError("Loss needs to be defined.")

    def _compute_domain_dist(self, x, domain_label):
        raise NotImplementedError("You need to implement a domain distance measure.")
        
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
        for domain_label_ in dataset.domain_to_idx.values():
            if domain_label_ != target_label:
                self.classifiers[domain_label_] = task_classifier
        self.k_moment = k_moment

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        x = self.forward(x)
        moment_loss = self._compute_domain_dist(x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tar_idx = torch.where(domain_labels == self.target_label)
        cls_loss, ok_src = self._compute_cls_loss(x[src_idx], y[src_idx], domain_labels[src_idx])
        if len(tar_idx) > 0:
            y_tar_hat = average_cls_output(x[tar_idx], self.classifiers)
            _, ok_tgt = losses.cross_entropy_logits(y_tar_hat, y[tar_idx])
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
            n_iter = 0.0
            for domain_label_ in unique_domain_:
                if domain_label_ == self.target_label:
                    continue
                domain_idx = torch.where(domain_labels == domain_label_)
                cls_output = self.classifiers[domain_label_](x[domain_idx])
                loss_cls_, ok_src_ = losses.cross_entropy_logits(cls_output, y[domain_idx])
                cls_loss += loss_cls_
                ok_src.append(ok_src_)
                n_iter += 1.0
            cls_loss = cls_loss / n_iter
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
        for i in range(self.k_moment - 1):
            moment_loss += _moment_k(x, domain_labels, i + 2)

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
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        x = self.forward(x)
        loss_dist = self._compute_domain_dist(x, domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tar_idx = torch.where(domain_labels == self.target_label)
        cls_output = self.classifier(x)
        loss_cls, ok_src = losses.cross_entropy_logits(cls_output[src_idx], y[src_idx])
        _, ok_tgt = losses.cross_entropy_logits(cls_output[tar_idx], y[tar_idx])

        task_loss = loss_cls
        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_dist": loss_dist,
        }

        return task_loss, loss_dist, log_metrics

    def _compute_domain_dist(self, x, domain_labels):
        if self.kernel == "linear":
            kx = torch.mm(x, x)
        else:
            raise ValueError("Other kernels have not been implemented yet!")
        ky = torch.mm(domain_labels, domain_labels)
        return losses.hsic(kx, ky)

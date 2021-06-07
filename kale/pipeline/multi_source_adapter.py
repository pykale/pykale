# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

from typing import Any, Dict

import torch

import kale.predict.losses as losses

# from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.pipeline.domain_adapter import BaseAdaptTrainer, get_aggregated_metrics


def _moment_k(x: torch.Tensor, domain_labels: torch.Tensor, k_order=1):
    unique_domain_ = torch.unique(domain_labels)
    n_unique_domain_ = len(unique_domain_)
    x_order_k = []
    for domain_label_ in unique_domain_:
        domain_idx = torch.where(domain_labels == domain_label_)
        if k_order == 1:
            x_order_k.append(x[domain_idx] - x[domain_idx].mean(0))
        else:
            x_order_k.append((x[domain_idx] ** k_order).mean(0))
    moment_ = 0
    for i in range(n_unique_domain_):
        for j in range(i + 1, n_unique_domain_):
            moment_ += losses.euclidean(x_order_k[i], x_order_k[j])

    return moment_


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
        super().__init__(dataset, feature_extractor, **base_params)
        if target_label not in dataset.domain_to_idx.values:
            raise ValueError("The given target label %s not in the given dataset! The available domain labels are %s"
                             % dataset.domain_to_idx.values)
        self._task_classifier = task_classifier
        
    def _compute_domain_dist(self, x, domain_label):
        raise NotImplementedError("You need to implement a domain distance measure.")
    
    def _compute_cls_loss(self, x, y, domain_labels):
        raise NotImplementedError("You need to implement a classification loss.")
        
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
        k_moment: int = 1,
        **base_params,
    ):
        super().__init__(dataset, feature_extractor, **base_params)

        self.classifier: Dict[int, Any] = {}
        self._task_classifier = task_classifier
        if target_label not in dataset.domain_to_idx.values:
            raise ValueError("The given target label %s not in the given dataset! The available domain labels are %s"
                             % dataset.domain_to_idx.values)
        self.target_label = target_label
        self.k_moment = k_moment

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        return x

    def compute_loss(self, batch, split_name="V"):
        x, y, domain_labels = batch
        x = self.forward(x)
        moment_loss = self._compute_k_moment(x, domain_labels)
        # unique_domain_ = torch.unique(domain_labels)
        src_idx = torch.where(domain_labels != self.target_label)
        tar_idx = torch.where(domain_labels == self.target_label)
        cls_loss, ok_src = self._compute_cls_loss(x[src_idx], y[src_idx], domain_labels[src_idx])
        y_tar_hat = average_cls_output(x[tar_idx], self.classifier)
        _, ok_tgt = losses.cross_entropy_logits(y_tar_hat, y[tar_idx])

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
            unique_domain_ = torch.unique(domain_labels)
            cls_loss = 0.0
            ok_src = 0.0
            n_iter = 0.0
            for domain_label_ in unique_domain_:
                if domain_label_ == self.target_label:
                    continue
                domain_idx = torch.where(domain_labels == domain_label_)
                if domain_label_ not in self.classifier:
                    self.classifier[domain_label_] = self._task_classifier
                cls_output = self.classifier[domain_label_](x[domain_idx])
                loss_cls_, ok_src_ = losses.cross_entropy_logits(cls_output, y)
                cls_loss += loss_cls_
                ok_src += ok_src_
                n_iter += 1.0
            cls_loss = cls_loss / n_iter
            return cls_loss, ok_src

    def _compute_domain_dist(self, x, domain_label):
        # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))

        moment_loss = _moment_k(x, domain_label, 1)

        # print(reg_info)
        for i in range(self.k_moment - 1):
            moment_loss += _moment_k(x, domain_label, i + 2)

        return moment_loss

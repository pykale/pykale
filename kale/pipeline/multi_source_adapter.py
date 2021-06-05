# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import pytorch_lightning as pl
import torch
from typing import Dict

import kale.predict.losses as losses
from kale.pipeline.domain_adapter import BaseAdaptTrainer


class M3SDATrainer(BaseAdaptTrainer):
    def __init__(self, dataset, feature_extractor, task_classifier, kernel_mul=2.0, kernel_num=5, **base_params):
        super().__init__(dataset, feature_extractor, **base_params)

        self.classifier: Dict[int, task_classifier] = {}
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num
        self.target_label = dataset.target_label

    def forward(self, x):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        class_output = torch.cat([self.classifier[domain_label_](x) for domain_label_ in self.classifier], 1)
        return x, class_output

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):


    def compute_loss(self, batch, split_name="V"):


        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": mmd,
        }

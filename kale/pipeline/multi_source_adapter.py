# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk/sz144@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import pytorch_lightning as pl
import torch

import kale.predict.losses as losses
from kale.pipeline.domain_adapter import BaseAdaptTrainer


class M3SDATrainer(BaseAdaptTrainer):
    def __init__(self, dataset, feature_extractor, task_classifier, n_source, kernel_mul=2.0, kernel_num=5, **base_params):
        super().__init__(dataset, feature_extractor, **base_params)

        self.n_source = n_source
        self.classifier = [task_classifier for i in range(n_source)]
        self._kernel_mul = kernel_mul
        self._kernel_num = kernel_num

    def forward(self, x, domain_label):
        if self.feat is not None:
            x = self.feat(x)
        x = x.view(x.size(0), -1)
        class_output = self.classifier[domain_label](x)
        return x, class_output

    def _compute_mmd(self, phi_s, phi_t, y_hat, y_t_hat):


    def compute_loss(self, batch, split_name="V"):


        log_metrics = {
            f"{split_name}_source_acc": ok_src,
            f"{split_name}_target_acc": ok_tgt,
            f"{split_name}_domain_acc": mmd,
        }

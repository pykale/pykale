# =============================================================================
# Author: Wenrui Fan, wenrui.fan@sheffield.ac.uk
# =============================================================================

"""ProtoNet trainer (pipelines)

This module contains the ProtoNet trainer class and its related functions.
It is used to train the ProtoNet model in N-way-k-shot problems. 

This module uses `PyTorch Lightning <https://github.com/Lightning-AI/lightning>`_ 
to standardize the flow.

This is a modified version of original prototypical networks for few-shot learning 
projects from https://github.com/jakesnell/prototypical-networks.
"""

from typing import Any
import yacs
import pytorch_lightning as pl
import torch
from kale.predict.losses import proto_loss


class ProtoNetTrainer(pl.LightningModule):
    """ProtoNet trainer class

    Args:
        cfg (yacs.config.CfgNode): Configurations of trainer.
        model (torch.nn.Module): A feature extractor without fc layer.
                                 Output 1-D vectors.
    """
    def __init__(self, 
                 cfg: yacs.config.CfgNode,
                 net: torch.nn.Module
                 ) -> None:
        super().__init__()

        self.cfg = cfg

        # model
        self.model = net

        # loss
        self.loss_train = proto_loss(
            n_ways=cfg.TRAIN.N_WAYS,
            k_query=cfg.TRAIN.K_QUERIES,
            device=cfg.DEVICE
        )
        self.loss_val = proto_loss(
            n_ways=cfg.VAL.N_WAYS,
            k_query=cfg.VAL.K_QUERIES,
            device=cfg.DEVICE
        )

    def forward(self, images, k_shots, n_ways) -> torch.Tensor:
        images = images.to(self.cfg.DEVICE)
        supports = images[0][0:k_shots]
        queries = images[0][k_shots:]
        for image in images[1:]:
            supports = torch.cat((supports, image[0:k_shots]), dim=0)
            queries = torch.cat((queries, image[k_shots:]), dim=0)
        feature_sup = self.model(supports).reshape(n_ways, k_shots, -1)
        feature_que = self.model(queries)
        return feature_sup, feature_que

    def compute_loss(self, feature_sup, feature_que, mode="train") -> Any:
        """
        Compute loss and accuracy. Here we use the same loss function for both
        training and validation, which is Euclidean distance related. 

        Args:
            feature_sup (torch.Tensor): Support features.
            feature_que (torch.Tensor): Query features.
            mode (str): Mode of the trainer, "train" or "val".

        Returns:
            loss (torch.Tensor): Loss value.
            return_dict (dict): Dictionary of loss and accuracy.
        """
        loss, acc = eval(f"self.loss_{mode}")(feature_sup, feature_que)
        return_dict = {"{}_loss".format(mode): loss.item(),
                       "{}_acc".format(mode): acc}
        return loss, return_dict

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step. Compute loss and accuracy, and log them by self.log_dict.
        For training, log on each step and each epoch. For validation and testing, 
        only log on each epoch. This way can avoid using on_training_epoch_end() 
        and on_validation_epoch_end().
        """
        images, _ = batch
        feature_sup, feature_que = self.forward(
            images,
            self.cfg.TRAIN.K_SHOTS,
            self.cfg.TRAIN.N_WAYS
        )
        loss, metrics = self.compute_loss(feature_sup,
                                          feature_que,
                                          mode="train")
        self.log_dict(metrics,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Compute and return the validation loss and metrics on one step."""
        images, _ = batch
        feature_sup, feature_que = self.forward(
            images,
            self.cfg.VAL.K_SHOTS,
            self.cfg.VAL.N_WAYS
        )
        _, metrics = self.compute_loss(feature_sup, feature_que, mode="val")
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)
        
    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Compute and return the test loss and metrics on one step."""
        images, _ = batch
        feature_sup, feature_que = self.forward(
            images,
            self.cfg.VAL.K_SHOTS,
            self.cfg.VAL.N_WAYS
        )
        _, metrics = self.compute_loss(feature_sup, feature_que, mode="val")
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training. Can be modified to support different
        optimizers from torch.optim.
        """
        optimizer = eval(f"torch.optim.{self.cfg.TRAIN.OPTIMIZER}")(
            self.model.parameters(),
            lr=self.cfg.TRAIN.LEARNING_RATE
            )
        return optimizer

    def get_metrics(self, trainer, model) -> dict:
        """Remove v_num from the progress bar."""
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

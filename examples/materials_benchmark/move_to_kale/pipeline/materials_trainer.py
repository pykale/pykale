import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
# from pytorch_lightning import Trainer
# from sklearn.metrics import accuracy_score

# from cgcnn_train_bg import train


class MaterialsTrainer(pl.LightningModule):
    """    A PyTorch Lightning module for training materials models.
    This class encapsulates the model, optimizer, and training logic.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (dict): A dictionary containing optimizer type and parameters.
        max_epochs (int): The maximum number of epochs for training.
        init_lr (float): Initial learning rate for the optimizer.
        adapt_lr (bool): Whether to adapt the learning rate during training.
        **kwargs: Additional keyword arguments for flexibility in configuration.
    """


    def __init__(self, model, optimizer, max_epochs, layer_freeze='all', init_lr=0.001, adapt_lr=False, **kwargs):
        super(MaterialsTrainer, self).__init__()
        self.model = model
        self._optimizer_params = optimizer
        self._max_epochs = max_epochs
        self._init_lr = init_lr
        self._adapt_lr = adapt_lr

    def forward(self, batch):

        return self.model(batch)

    def compute_loss(self, batch, split_name="val"):
        
        results = self.forward(batch)
    
        # If forward returns a single object (not tuple/list), wrap it so we can index it.
        if not isinstance(results, (tuple, list)):
            results = (results,)
        
        # Extract only the first item.
        output = results[0]

        target = batch.y.to(self.device)
        l1_loss = nn.L1Loss()
        loss = nn.MSELoss()(output, target)
        mae = l1_loss(output, target)
        mre = torch.mean(torch.abs(output - target) / (target + 1e-8))  # mean relative error
        # Calculate R² (Coefficient of Determination)
        ss_total = torch.sum((target - torch.mean(target)) ** 2)
        ss_residual = torch.sum((target - output) ** 2)
        r2 = 1 - ss_residual / (ss_total + 1e-8)

        log_metrics = {
            f"{split_name}_loss": loss,
            f"{split_name}_mae": mae,
            f"{split_name}_mre": mre,
            f"{split_name}_r2": r2,
        }
        return loss, log_metrics

    def configure_optimizers(self):
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr)
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self._init_lr,
                **self._optimizer_params["optim_params"],
            )

            if self._adapt_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epochs)
                return [optimizer], [scheduler]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}.")

    def training_step(self, train_batch, batch_idx):
        loss, metrics = self.compute_loss(train_batch, split_name="train")
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]['lr']
        metrics['learning_rate'] = current_lr
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True, batch_size=train_batch.batch_size)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        loss, metrics = self.compute_loss(valid_batch, split_name="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=valid_batch.batch_size)

    def test_step(self, test_batch, batch_idx):
        loss, metrics = self.compute_loss(test_batch, split_name="test")
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=test_batch.batch_size)
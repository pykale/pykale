from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseTrainer(pl.LightningModule):
    """
    Base class for classification models, based on pytorch lightning wrapper.
    If you inherit from this class, a forward pass function must be implemented.

    Args:
        optimizer (dict): optimizer parameters.
        max_epochs (int): maximum number of epochs.
        init_lr (float): initial learning rate. (default: 0.001)
        adapt_lr (bool): adapt learning rate during training. (default: :obj:`False`)
    """

    def __init__(self, optimizer, max_epochs, init_lr=0.001, adapt_lr=False):
        super(BaseTrainer, self).__init__()
        self._init_lr = init_lr
        self._optimizer_params = optimizer
        self._adapt_lr = adapt_lr
        self._max_epochs = max_epochs

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        if self._optimizer_params is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr)
            return [optimizer]
        if self._optimizer_params["type"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)
            return [optimizer]
        if self._optimizer_params["type"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)

            if self._adapt_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epochs, last_epoch=-1)
                return [optimizer], [scheduler]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}")

    def forward(self, x):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x, y = train_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.logger.log_metrics({"train_loss": loss}, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x, y = val_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x, y = test_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss


class CNNTransformerTrainer(BaseTrainer):

    """Pytorch Lightning trainer for cifar-cnntransformer
    Args:
        model (torch.nn.Sequential): model according to the config
        optimizer (dict): parameters of the model
        cfg (CfgNode): hyperparameters from configure file
    """

    def __init__(self, model, optimizer, cfg):
        super().__init__(optimizer, cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.BASE_LR)

        self.model = model
        self.optim = optimizer
        self.cfg = cfg

        self.loss_fn = nn.NLLLoss()
        self.train_acc, self.valid_acc = [], []
        self.best_valid_acc = 0

        self.ave_time = 0
        self.epochs = 1

    def forward(self, x):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y)
        _, predicted = y_pred.max(1)
        acc = (predicted == y).sum().item() / y.size(0)

        self.log("train_loss", loss)
        self.log("train_acc", acc * 100, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        _, predicted = y_pred.max(1)
        acc = (predicted == y).sum().item() / y.size(0)

        self.log("val_loss", loss)
        self.log("val_acc", acc * 100, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        _, predicted = y_pred.max(1)
        acc = (predicted == y).sum().item() / y.size(0)

        self.log("val_loss", loss)
        self.log("val_acc", acc * 100, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        c = self.cfg
        if c.SOLVER.WARMUP and self.epochs < c.SOLVER.WARMUP_EPOCHS:
            lr = c.SOLVER.BASE_LR * self.epochs / c.SOLVER.WARMUP_EPOCHS
        else:
            # normal (step) scheduling
            lr = c.SOLVER.BASE_LR
            for m_epoch in c.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= c.SOLVER.LR_GAMMA

        for param_group in self.optim["param_groups"]:

            param_group["lr"] = lr
            if "scaling" in param_group:
                param_group["lr"] *= param_group["scaling"]

        # set the optimizer
        train_params = self.optim["param_groups"][0]
        train_params_local = deepcopy(train_params)
        try:
            del train_params_local["lr"]
        except KeyError:
            pass

        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim["param_groups"][0]["lr"],)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, last_epoch=-1)
        return [optimizer], [scheduler]


# class CNNTransformerTrainer(pl.LightningModule):
#
#     """Pytorch Lightning trainer for cifar-cnntransformer
#     Args:
#         model (torch.nn.Sequential): model according to the config
#         optim (dict): parameters of the model
#         cfg: A YACS config object
#     """
#
#     def __init__(self, model, optim, cfg):
#         super().__init__()
#
#         self.model = model
#         self.optim = optim
#         self.cfg = cfg
#
#         self.loss_fn = nn.NLLLoss()
#         self.train_acc, self.valid_acc = [], []
#         self.best_valid_acc = 0
#
#         self.ave_time = 0
#         self.epochs = 1
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch):
#
#         x, y = batch
#         x, y = x.to(self.device), y.to(self.device)
#
#         outputs = self.model(x)
#
#         loss = self.loss_fn(outputs, y)
#         _, predicted = outputs.max(1)
#         acc = (predicted == y).sum().item() / y.size(0)
#
#         self.log("train_loss", loss)
#         self.log("train_acc", acc * 100, prog_bar=True)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         x, y = x.to(self.device), y.to(self.device)
#
#         outputs = self.model(x)
#         loss = self.loss_fn(outputs, y)
#         _, predicted = outputs.max(1)
#         acc = (predicted == y).sum().item() / y.size(0)
#
#         self.log("val_loss", loss)
#         self.log("val_acc", acc * 100, prog_bar=True)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         x, y = x.to(self.device), y.to(self.device)
#
#         outputs = self.model(x)
#         loss = self.loss_fn(outputs, y)
#         _, predicted = outputs.max(1)
#         acc = (predicted == y).sum().item() / y.size(0)
#
#         self.log("val_loss", loss)
#         self.log("val_acc", acc * 100, prog_bar=True)
#         return loss
#
#     def configure_optimizers(self):
#         c = self.cfg
#         if c.SOLVER.WARMUP and self.epochs < c.SOLVER.WARMUP_EPOCHS:
#             lr = c.SOLVER.BASE_LR * self.epochs / c.SOLVER.WARMUP_EPOCHS
#         else:
#             # normal (step) scheduling
#             lr = c.SOLVER.BASE_LR
#             for m_epoch in c.SOLVER.LR_MILESTONES:
#                 if self.epochs > m_epoch:
#                     lr *= c.SOLVER.LR_GAMMA
#
#         for param_group in self.optim["param_groups"]:
#
#             param_group["lr"] = lr
#             if "scaling" in param_group:
#                 param_group["lr"] *= param_group["scaling"]
#
#         # set the optimizer
#         train_params = self.optim["param_groups"][0]
#         train_params_local = deepcopy(train_params)
#         try:
#             del train_params_local["lr"]
#         except KeyError:
#             pass
#
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.optim["param_groups"][0]["lr"],)
#
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, last_epoch=-1)
#         return [optimizer], [scheduler]

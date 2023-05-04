# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haolin Wang, LWang0101@outlook.com
# =============================================================================


"""Classification systems (pipelines)

This module provides neural network (nn) trainers for developing classification task models. The BaseNNTrainer
defines the required fundamental functions and structures, such as the optimizer, learning rate scheduler,
training/validation/testing procedure, workflow, etc. The BaseNNTrainer is inherited to construct specialized trainers.

The structure and workflow of BaseNNTrainer is consistent with `kale.pipeline.domain_adapter.BaseAdaptTrainer`

This module uses `PyTorch Lightning <https://github.com/Lightning-AI/lightning>`_ to standardize the flow.
"""


import pytorch_lightning as pl
import torch

from kale.predict import losses


class BaseNNTrainer(pl.LightningModule):
    """Base class for classification models using neural network, based on PyTorch Lightning wrapper. The forward pass
    and loss computation must be implemented if new trainers inherit from this class. The basic workflow is defined
    in this class as follows. Every training/validation/testing procedure will call `compute_loss()` to compute the
    loss and log the output metrics. The `compute_loss()` function will call `forward()` to generate the output feature
    using the neural networks.

    Args:
        optimizer (dict, None): optimizer parameters.
        max_epochs (int): maximum number of epochs.
        init_lr (float): initial learning rate. Defaults to 0.001.
        adapt_lr (bool): whether to use the schedule for the learning rate. Defaults to False.
    """

    def __init__(self, optimizer, max_epochs, init_lr=0.001, adapt_lr=False):
        super(BaseNNTrainer, self).__init__()
        self._optimizer_params = optimizer
        self._max_epochs = max_epochs
        self._init_lr = init_lr
        self._adapt_lr = adapt_lr

    def forward(self, x):
        """Override this function to define the forward pass. Normally includes feature extraction and classification
        and be called in `compute_loss()`.
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def compute_loss(self, batch, split_name="valid"):
        """Compute loss for a given batch.

        Args:
            batch (tuple): batches returned by dataloader.
            split_name (str, optional): learning stage (one of ["train", "valid", "test"]).
                Defaults to "valid" for validation. "train" is for training and "test" for testing. This is currently
                used only for naming the metrics used for logging.

        Returns:
            loss (torch.Tensor): loss value.
            log_metrics (dict): dictionary of metrics to be logged. This is needed when using PyKale logging, but not
                mandatory when using PyTorch Lightning logging.
        """
        raise NotImplementedError("Loss function needs to be defined.")

    def configure_optimizers(self):
        """Default optimizer configuration. Set Adam to the default and provide SGD with cosine annealing.
        If other optimizers are needed, please override this function.
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
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epochs)
                return [optimizer], [scheduler]
            return [optimizer]
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}.")

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """Compute and return the training loss and metrics on one step. loss is to store the loss value. log_metrics
        is to store the metrics to be logged, including loss, top1 and/or top5 accuracies.

        Use self.log_dict(log_metrics, on_step, on_epoch, logger) to log the metrics on each step and each epoch. For
        training, log on each step and each epoch. For validation and testing, only log on each epoch. This way can
        avoid using on_training_epoch_end() and on_validation_epoch_end().
        """
        loss, log_metrics = self.compute_loss(train_batch, split_name="train")
        self.log_dict(log_metrics, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, valid_batch, batch_idx) -> None:
        """Compute and return the validation loss and metrics on one step."""
        loss, log_metrics = self.compute_loss(valid_batch, split_name="valid")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, logger=True)

    def test_step(self, test_batch, batch_idx) -> None:
        """Compute and return the testing loss and metrics on one step."""
        loss, log_metrics = self.compute_loss(test_batch, split_name="test")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, logger=True)


class CNNTransformerTrainer(BaseNNTrainer):
    """PyTorch Lightning trainer for cnntransformer.

    Args:
        feature_extractor (torch.nn.Sequential, optional): the feature extractor network.
        optimizer (dict): optimizer parameters.
        lr_milestones (list): list of epoch indices. Must be increasing.
        lr_gamma (float): multiplicative factor of learning rate decay.
    """

    def __init__(self, feature_extractor, task_classifier, lr_milestones, lr_gamma, **kwargs):
        super().__init__(**kwargs)
        self.feat = feature_extractor
        self.classifier = task_classifier
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma

    def forward(self, x):
        """Forward pass for the model with a feature extractor and a classifier."""
        x = self.feat(x)
        output = self.classifier(x)
        return output

    def compute_loss(self, batch, split_name="valid"):
        """Compute loss, top1 and top5 accuracy for a given batch."""
        x, y = batch
        y_hat = self.forward(x)

        loss, _ = losses.cross_entropy_logits(y_hat, y)
        top1, top5 = losses.topk_accuracy(y_hat, y, topk=(1, 5))
        top1 = top1.double().mean()
        top5 = top5.double().mean()

        log_metrics = {
            f"{split_name}_loss": loss,
            f"{split_name}_top1_acc": top1,
            f"{split_name}_top5_acc": top5,
        }

        return loss, log_metrics

    def configure_optimizers(self):
        """Set up an SGD optimizer and multistep learning rate scheduler. When self._adapt_lr is True, the learning
        rate will be decayed by self.lr_gamma every step in milestones.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)
        if self._adapt_lr:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma
            )
            return [optimizer], [scheduler]
        return [optimizer]

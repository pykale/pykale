# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haolin Wang, LWang0101@outlook.com
# =============================================================================

"""Classification of data
The main structure is borrowed from kale.pipeline.domain_adapter.
"""


import pytorch_lightning as pl
import torch

from kale.predict import losses


class BaseTrainer(pl.LightningModule):
    """
    Base class for classification models, based on pytorch lightning wrapper.
    If you inherit from this class, a forward pass function must be implemented.
    The structure is borrowed from kale.pipeline.domain_adapter.BaseAdaptTrainer.

    Args:
        optimizer (dict): optimizer parameters.
        max_epochs (int): maximum number of epochs.
        init_lr (float): initial learning rate. (default: 0.001)
        adapt_lr (bool): adapt learning rate during training. (default: :obj:`False`)
    """

    def __init__(self, optimizer, max_epochs, init_lr=0.001, adapt_lr=False):
        super(BaseTrainer, self).__init__()
        self._optimizer_params = optimizer
        self._max_epochs = max_epochs
        self._init_lr = init_lr
        self._adapt_lr = adapt_lr

    def configure_optimizers(self):
        """
        Default optimizer configuration. Config Adam as default optimizer and provide SGD with cosine annealing.
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
        raise NotImplementedError(f"Unknown optimizer type {self._optimizer_params['type']}")

    def forward(self, x):
        """
        Override this function to define the forward pass. Same as :meth:`torch.nn.Module.forward()`.
        Normally includes feature extraction and classification and be called in :meth:`compute_loss()`.
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def compute_loss(self, batch, split_name="valid"):
        """
        Compute loss for a given batch.

        Args:
            batch (tuple): batches returned by dataloader.
            split_name (str, optional): learning stage (one of ["train", "valid", "test"]).
                Defaults to "valid" for validation. "train" is for training and "test" for testing.
                This is currently used only for naming the metrics used for logging.

        Returns:
            loss (torch.Tensor): loss value.
            log_metrics (dict): dictionary of metrics to be logged. This is needed when using PyKale logging,
                but not mandatory when using PyTorch lightning logging.
        """
        raise NotImplementedError("Loss function needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        loss = self.compute_loss(train_batch, split_name="train")
        return loss

    def validation_step(self, valid_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        loss = self.compute_loss(valid_batch, split_name="valid")
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        loss = self.compute_loss(test_batch, split_name="test")
        return loss


class CNNTransformerTrainer(BaseTrainer):

    """Pytorch Lightning trainer for cifar-cnntransformer
    Args:
        feature_extractor (torch.nn.Sequential): model according to the config
        optimizer (dict): parameters of the model
        lr_milestones (list): list of epoch indices. Must be increasing.
        lr_gamma (float): multiplicative factor of learning rate decay.
    """

    def __init__(
        self, feature_extractor, task_classifier, lr_milestones, lr_gamma, **kwargs,
    ):
        super().__init__(**kwargs)
        self.feat = feature_extractor
        self.classifier = task_classifier
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma

    def forward(self, x):
        x = self.feat(x)
        output = self.classifier(x)
        return output

    def compute_loss(self, batch, split_name="valid"):
        """
        Compute loss, top1 and top5 accuracy for a given batch.
        """
        x, y = batch
        y_hat = self.forward(x)

        loss, _ = losses.cross_entropy_logits(y_hat, y)
        top1, top5 = losses.topk_accuracy(y_hat, y, topk=(1, 5))
        top1 = top1.double().mean()
        top5 = top5.double().mean()

        self.log(f"{split_name}_loss", loss)
        self.log(f"{split_name}_top1_acc", top1)
        self.log(f"{split_name}_top5_acc", top5)
        return loss

    def configure_optimizers(self):
        """
        Set up an SGD optimizer and multistep learning rate scheduler.
        When self._adapt_lr is True, the learning rate will be decayed by self.lr_gamma every step in milestones.
        """

        optimizer = torch.optim.SGD(self.parameters(), lr=self._init_lr, **self._optimizer_params["optim_params"],)
        if self._adapt_lr:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma
            )
            return [optimizer], [scheduler]
        return [optimizer]

import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class BaseDTATrainer(pl.LightningModule):
    """
    Base class for all drug target encoder-decoder architecture models, which is based on pytorch lightning wrapper,
    for more details about pytorch lightning, please check https://github.com/PyTorchLightning/pytorch-lightning.
    If you inherit from this class, a forward pass function must be implemented.

    Args:
        drug_encoder: drug information encoder.
        target_encoder: target information encoder.
        decoder: drug-target representations decoder.
        lr: learning rate.
    """

    def __init__(self, drug_encoder, target_encoder, decoder, lr, **kwargs):
        super(BaseDTATrainer, self).__init__()
        self.drug_encoder = drug_encoder
        self.target_encoder = target_encoder
        self.decoder = decoder
        self.lr = lr
        if kwargs:
            self.save_hyperparameters(kwargs)

    def configure_optimizers(self):
        """
        Config adam as default optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x_drug, x_target):
        """
        Same as :meth:`torch.nn.Module.forward()`
        """
        raise NotImplementedError("Forward pass needs to be defined.")

    def training_step(self, train_batch, batch_idx):
        """
        Compute and return the training loss on one step
        """
        x_drug, x_target, y = train_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.logger.log_metrics({"train_step_loss": loss}, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Compute and return the validation loss on one step
        """
        x_drug, x_target, y = val_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Compute and return the test loss on one step
        """
        x_drug, x_target, y = test_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss


class DeepDTATrainer(BaseDTATrainer):
    """
    An implementation of DeepDTA model based on BaseDTATrainer.
    Args:
        drug_encoder: drug CNN encoder.
        target_encoder: target CNN encoder.
        decoder: drug-target MLP decoder.
        lr: learning rate.
    """

    def __init__(self, drug_encoder, target_encoder, decoder, lr, **kwargs):
        super().__init__(drug_encoder, target_encoder, decoder, lr, **kwargs)

    def forward(self, x_drug, x_target):
        """
        Forward propagation in DeepDTA architecture.

        Args:
            x_drug: drug sequence encoding.
            x_target: target protein sequence encoding.
        """
        drug_emb = self.drug_encoder(x_drug)
        target_emb = self.target_encoder(x_target)
        comb_emb = torch.cat((drug_emb, target_emb), dim=1)
        output = self.decoder(comb_emb)
        return output

    def validation_step(self, val_batch, batch_idx):
        x_drug, x_target, y = val_batch
        y_pred = self(x_drug, x_target)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

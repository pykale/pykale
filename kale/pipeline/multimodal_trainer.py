# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
multimodal_trainer.py

Contains all trainer classes for multimodal training, including MultimodalTriStreamVAETrainer for joint optimization and validation of tri-stream MVAE pre-training with image and signal modalities.
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader

from kale.evaluate.metrics import signal_image_elbo_loss
from kale.predict.decode import SignalImageFineTuningClassifier


class SignalImageTriStreamVAETrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for tri-stream multimodal variational autoencoders (MVAE) with image and 1D signal modalities.

    This class manages the training and validation steps for a MVAE, supporting separate and joint reconstruction
    loss computation for each modality and the fused latent distribution. It is agnostic to input domains and supports
    both training and validation via PyTorch Lightning’s workflow.

    Args:
        model (nn.Module): The multimodal VAE model to be trained.
        train_dataset (Dataset): PyTorch dataset providing training samples (image, signal) pairs.
        val_dataset (Dataset): PyTorch dataset for validation (optional; can be None).
        batch_size (int, optional): Batch size for training/validation. Default is 32.
        num_workers (int, optional): Number of DataLoader worker processes. Default is 4.
        lambda_image (float, optional): Weight for the image reconstruction loss. Default is 1.0.
        lambda_signal (float, optional): Weight for the signal reconstruction loss. Default is 10.0.
        lr (float, optional): Learning rate for Adam optimizer. Default is 1e-3.
        annealing_epochs (int, optional): Number of epochs for KL annealing. Default is 50.
        scale_factor (float, optional): Overall scaling factor for reconstruction loss. Default is 1e-4.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=4,
        lambda_image=1.0,
        lambda_signal=10.0,
        lr=1e-3,
        annealing_epochs=50,
        scale_factor=1e-4,
    ):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lambda_image = lambda_image
        self.lambda_signal = lambda_signal
        self.lr = lr
        self.annealing_epochs = annealing_epochs
        self.scale_factor = scale_factor
        self.save_hyperparameters(ignore=["model", "train_dataset", "val_dataset"])

    def forward(self, image=None, signal=None):
        """
        Forward pass through the multimodal VAE model.

        Args:
            image (Tensor, optional): Batch of image modality inputs.
            signal (Tensor, optional): Batch of signal modality inputs.

        Returns:
            Tuple: Model outputs (image reconstruction, signal reconstruction, mu, logvar).
        """
        return self.model(image, signal)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step with reconstruction and KL losses for all input combinations.

        Args:
            batch (tuple): Tuple of (signal, image) tensors.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss for the current batch.
        """
        signal, image = batch
        signal = signal.to(self.device)
        image = image.to(self.device)

        epoch = self.current_epoch
        annealing_factor = min(epoch / self.annealing_epochs, 1.0)
        batch_size = signal.size(0)

        # Forward passes for joint, image-only, and signal-only
        outputs = {
            "joint": self.model(image, signal),
            "image_only": self.model(image=image),
            "signal_only": self.model(signal=signal),
        }

        # Corresponding targets for each loss type
        loss_args = {
            "joint": {
                "recon_image": outputs["joint"][0],
                "target_image": image,
                "recon_signal": outputs["joint"][1],
                "target_signal": signal,
                "mu": outputs["joint"][2],
                "logvar": outputs["joint"][3],
            },
            "image_only": {
                "recon_image": outputs["image_only"][0],
                "target_image": image,
                "recon_signal": None,
                "target_signal": None,
                "mu": outputs["image_only"][2],
                "logvar": outputs["image_only"][3],
            },
            "signal_only": {
                "recon_image": None,
                "target_image": None,
                "recon_signal": outputs["signal_only"][1],
                "target_signal": signal,
                "mu": outputs["signal_only"][2],
                "logvar": outputs["signal_only"][3],
            },
        }

        # Shared loss arguments
        shared_kwargs = {
            "lambda_image": self.lambda_image,
            "lambda_signal": self.lambda_signal,
            "annealing_factor": annealing_factor,
            "scale_factor": self.scale_factor,
        }

        # Compute all three losses using a loop
        total_loss = 0.0
        for mode in ["joint", "image_only", "signal_only"]:
            args = loss_args[mode]
            total_loss += signal_image_elbo_loss(
                args["recon_image"],
                args["target_image"],
                args["recon_signal"],
                args["target_signal"],
                args["mu"],
                args["logvar"],
                **shared_kwargs,
            )

        train_loss = total_loss / batch_size
        self.log("train_loss", train_loss, prog_bar=True, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step, mirroring the training loss calculations.

        Args:
            batch (tuple): Tuple of (signal, image) tensors.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Validation loss for the current batch.
        """
        signal, image = batch
        signal = signal.to(self.device)
        image = image.to(self.device)

        epoch = self.current_epoch
        annealing_factor = min(epoch / self.annealing_epochs, 1.0)
        batch_size = signal.size(0)

        # Forward passes for joint, image-only, and signal-only
        outputs = {
            "joint": self.model(image, signal),
            "image_only": self.model(image=image),
            "signal_only": self.model(signal=signal),
        }

        # Prepare inputs for loss calculation
        loss_args = {
            "joint": {
                "recon_image": outputs["joint"][0],
                "target_image": image,
                "recon_signal": outputs["joint"][1],
                "target_signal": signal,
                "mu": outputs["joint"][2],
                "logvar": outputs["joint"][3],
            },
            "image_only": {
                "recon_image": outputs["image_only"][0],
                "target_image": image,
                "recon_signal": None,
                "target_signal": None,
                "mu": outputs["image_only"][2],
                "logvar": outputs["image_only"][3],
            },
            "signal_only": {
                "recon_image": None,
                "target_image": None,
                "recon_signal": outputs["signal_only"][1],
                "target_signal": signal,
                "mu": outputs["signal_only"][2],
                "logvar": outputs["signal_only"][3],
            },
        }

        shared_kwargs = {
            "lambda_image": self.lambda_image,
            "lambda_signal": self.lambda_signal,
            "annealing_factor": annealing_factor,
            "scale_factor": self.scale_factor,
        }

        # Compute the three validation losses
        total_loss = 0.0
        for mode in ["joint", "image_only", "signal_only"]:
            args = loss_args[mode]
            total_loss += signal_image_elbo_loss(
                args["recon_image"],
                args["target_image"],
                args["recon_signal"],
                args["target_signal"],
                args["mu"],
                args["logvar"],
                **shared_kwargs,
            )

        val_loss = total_loss / batch_size
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """
        Configures the optimizer (Adam) for model training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation set.
        """
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            return None


class SignalImageFineTuningTrainer(pl.LightningModule):
    """
    SignalImageFineTuningTrainer trains and validates a supervised classifier built on top of frozen encoders from a pre-trained multimodal model.

    This LightningModule wraps a MultimodalClassifier (which fuses representations from image and signal modalities) and supports
    cross-entropy loss optimization, as well as batch-aggregated computation of validation accuracy, ROC-AUC, and MCC.
    It is compatible with any dataset yielding (image, signal, label) batches and is suitable for transfer learning scenarios
    where the encoder weights are fixed.

    Args:
        pretrained_model (kale.embed.multimodal_encoder.SignalImageVAE): Pre-trained multimodal model providing `image_encoder` and `signal_encoder` attributes.
        num_classes (int, optional): Number of classes for classification. Default is 2.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.

    Forward Input:
        image (Tensor): Image modality input.
        signal (Tensor): Signal modality input.

    Forward Output:
        logits (Tensor): Raw classification scores before softmax.

    Example:
        model = SignalImageFineTuningTrainer(pretrained_model, num_classes, hidden_dim)
        logits = model(images, signals)
    """

    def __init__(self, pretrained_model, num_classes=2, lr=1e-3, hidden_dim=128):
        super().__init__()
        self.model = SignalImageFineTuningClassifier(pretrained_model, num_classes, hidden_dim)
        self.lr = lr
        self.num_classes = num_classes
        self.validation_step_outputs = []
        self.train_losses = []
        self.val_losses = []

        # TorchMetrics (binary since num_classes == 2)
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    def forward(self, image, signal):
        return self.model(image, signal)

    def training_step(self, batch, batch_idx):
        image, signal, labels = batch
        logits = self(image, signal)
        loss = F.cross_entropy(logits, labels)
        self.train_losses.append(loss.detach().cpu())
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        if self.train_losses:
            avg_loss = torch.stack(self.train_losses).mean().item()
            self.log("train_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
            self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        image, signal, labels = batch
        logits = self(image, signal)
        loss = F.cross_entropy(logits, labels)
        self.val_losses.append(loss.detach())

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        self.val_accuracy.update(preds, labels)
        self.val_auc.update(probs, labels)
        self.val_mcc.update(preds, labels)

    def on_validation_epoch_end(self):
        if self.val_losses:
            avg_loss = torch.stack(self.val_losses).mean().item()
            self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True)
            self.val_losses.clear()

        self.log("val_acc", self.val_accuracy.compute(), prog_bar=True, on_epoch=True)
        self.log("val_auroc", self.val_auc.compute(), prog_bar=True, on_epoch=True)
        self.log("val_mcc", self.val_mcc.compute(), prog_bar=True, on_epoch=True)

        self.val_accuracy.reset()
        self.val_auc.reset()
        self.val_mcc.reset()

    def configure_optimizers(self):
        """
        Configures the Adam optimizer for training the classifier.
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
multimodal_trainer.py

Contains all trainer classes for multimodal training, including MultimodalTriStreamVAETrainer for joint optimization and validation of tri-stream MVAE pre-training with image and signal modalities.
"""
import torch
import pytorch_lightning as pl
from kale.evaluate.metrics import elbo_loss


class MultimodalTriStreamVAETrainer(pl.LightningModule):
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
        self.save_hyperparameters(ignore=['model', 'train_dataset', 'val_dataset'])

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
        recon_image_joint, recon_signal_joint, mu_joint, logvar_joint = self.model(image, signal)
        recon_image_only, _, mu_image, logvar_image = self.model(image=image)
        _, recon_signal_only, mu_signal, logvar_signal = self.model(signal=signal)
        batch_size = signal.size(0)
        joint_loss = elbo_loss(
            recon_image_joint, image, recon_signal_joint, signal, mu_joint, logvar_joint,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        image_loss = elbo_loss(
            recon_image_only, image, None, None, mu_image, logvar_image,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        signal_loss = elbo_loss(
            None, None, recon_signal_only, signal, mu_signal, logvar_signal,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        train_loss = (joint_loss + image_loss + signal_loss) / batch_size
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
        recon_image_joint, recon_signal_joint, mu_joint, logvar_joint = self.model(image, signal)
        recon_image_only, _, mu_image, logvar_image = self.model(image=image)
        _, recon_signal_only, mu_signal, logvar_signal = self.model(signal=signal)
        batch_size = signal.size(0)
        joint_loss = elbo_loss(
            recon_image_joint, image, recon_signal_joint, signal, mu_joint, logvar_joint,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        image_loss = elbo_loss(
            recon_image_only, image, None, None, mu_image, logvar_image,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        signal_loss = elbo_loss(
            None, None, recon_signal_only, signal, mu_signal, logvar_signal,
            self.lambda_image, self.lambda_signal, annealing_factor, self.scale_factor
        )
        val_loss = (joint_loss + image_loss + signal_loss) / batch_size
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """
        Configures the optimizer (Adam) for model training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

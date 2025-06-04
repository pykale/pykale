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
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

from kale.evaluate.metrics import elbo_loss
from kale.predict.decode import MultimodalClassifier


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
        recon_image_joint, recon_signal_joint, mu_joint, logvar_joint = self.model(image, signal)
        recon_image_only, _, mu_image, logvar_image = self.model(image=image)
        _, recon_signal_only, mu_signal, logvar_signal = self.model(signal=signal)
        batch_size = signal.size(0)
        joint_loss = elbo_loss(
            recon_image_joint,
            image,
            recon_signal_joint,
            signal,
            mu_joint,
            logvar_joint,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
        )
        image_loss = elbo_loss(
            recon_image_only,
            image,
            None,
            None,
            mu_image,
            logvar_image,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
        )
        signal_loss = elbo_loss(
            None,
            None,
            recon_signal_only,
            signal,
            mu_signal,
            logvar_signal,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
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
            recon_image_joint,
            image,
            recon_signal_joint,
            signal,
            mu_joint,
            logvar_joint,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
        )
        image_loss = elbo_loss(
            recon_image_only,
            image,
            None,
            None,
            mu_image,
            logvar_image,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
        )
        signal_loss = elbo_loss(
            None,
            None,
            recon_signal_only,
            signal,
            mu_signal,
            logvar_signal,
            self.lambda_image,
            self.lambda_signal,
            annealing_factor,
            self.scale_factor,
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


class MultimodalTrainer(pl.LightningModule):
    """
    MultimodalTrainer trains and validates a supervised classifier built on top of frozen encoders from a pre-trained multimodal model.

    This LightningModule wraps a MultimodalClassifier (which fuses representations from image and signal modalities) and supports
    cross-entropy loss optimization, as well as batch-aggregated computation of validation accuracy, ROC-AUC, and MCC.
    It is compatible with any dataset yielding (image, signal, label) batches and is suitable for transfer learning scenarios
    where the encoder weights are fixed.

    Args:
        pretrained_model (nn.Module): Pre-trained multimodal model providing `image_encoder` and `signal_encoder` attributes.
        num_classes (int, optional): Number of classes for classification. Default is 2.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.

    Forward Input:
        image (Tensor): Image modality input.
        signal (Tensor): Signal modality input.

    Forward Output:
        logits (Tensor): Raw classification scores before softmax.

    Example:
        model = MultimodalTrainer(pretrained_model, num_classes=3)
        logits = model(images, signals)
    """

    def __init__(self, pretrained_model, num_classes=2, lr=1e-3):
        super().__init__()
        self.model = MultimodalClassifier(pretrained_model, num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.validation_step_outputs = []
        self.train_losses = []
        self.val_losses = []

    def forward(self, image, signal):
        """
        Forward pass through the underlying classifier.
        """
        return self.model(image, signal)

    def training_step(self, batch, batch_idx):
        """
        Computes and logs the cross-entropy loss for a training batch.
        """
        image, signal, labels = batch
        logits = self(image, signal)
        loss = F.cross_entropy(logits, labels)
        self.train_losses.append(loss.detach().cpu())
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        """
        Logs the average training loss at the end of each epoch.
        """
        if self.train_losses:
            avg_loss = torch.stack(self.train_losses).mean().item()
            self.log("train_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
            self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        """
        Collects predictions, probabilities, and ground truths for validation metrics computation.
        """
        image, signal, labels = batch
        logits = self(image, signal)
        loss = F.cross_entropy(logits, labels)
        self.val_losses.append(loss.detach().cpu())
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        labels_cpu = labels.detach().cpu()
        preds_cpu = preds.detach().cpu()
        self.validation_step_outputs.append((labels_cpu, preds_cpu, probs))

    def on_validation_epoch_end(self):
        """
        Computes and logs validation loss, accuracy, ROC-AUC, and MCC at the end of each validation epoch.
        """
        if self.val_losses:
            avg_loss = torch.stack(self.val_losses).mean().item()
            self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True)
            self.val_losses.clear()

        labels_all, preds_all, probs_all = [], [], []
        for labels, preds, probs in self.validation_step_outputs:
            labels_all.append(labels)
            preds_all.append(preds)
            probs_all.append(probs)

        if labels_all:
            labels_all = torch.cat(labels_all).numpy()
            preds_all = torch.cat(preds_all).numpy()
            probs_all = torch.cat(probs_all).numpy()
            acc = accuracy_score(labels_all, preds_all)
            try:
                auc = roc_auc_score(labels_all, probs_all)
            except Exception:
                auc = float("nan")
            try:
                mcc = matthews_corrcoef(labels_all, preds_all)
            except Exception:
                mcc = float("nan")
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
            self.log("val_auc", auc, prog_bar=True, on_epoch=True)
            self.log("val_mcc", mcc, prog_bar=True, on_epoch=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures the Adam optimizer for training the classifier.
        """
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kale.pipeline.multimodal_trainer import SignalImageFineTuningTrainer, SignalImageTriStreamVAETrainer


class DummyMVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))  # <-- Fake parameter

    def forward(self, image=None, signal=None):
        batch_size = image.size(0) if image is not None else signal.size(0)
        recon_image = torch.zeros(batch_size, 4, 4)
        recon_signal = torch.zeros(batch_size, 10)
        mu = torch.zeros(batch_size, 6)
        logvar = torch.zeros(batch_size, 6)
        return recon_image, recon_signal, mu, logvar


class DummyDataset(Dataset):
    """
    Flexible dummy dataset for tests. If labels=True, returns (image, signal, label),
    otherwise returns (signal, image).
    """

    def __init__(self, num_samples=6, in_dim=5, num_classes=2, labels=False):
        self.num_samples = num_samples
        self.labels = labels
        self.images = torch.randn(num_samples, in_dim)
        self.signals = torch.randn(num_samples, in_dim)
        self.all_labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.labels:
            return self.images[idx], self.signals[idx], self.all_labels[idx]
        else:
            # For the VAE test, mimic (signal, image) tuple for compatibility.
            return self.signals[idx], self.images[idx]


# Dummy pre-trained model with frozen encoders
class DummyEncoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.linear = nn.Linear(5, latent_dim)

    def forward(self, x):
        return self.linear(x), torch.zeros(x.size(0), self.linear.out_features)


class DummyPretrainedModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.image_encoder = DummyEncoder(latent_dim)
        self.signal_encoder = DummyEncoder(latent_dim)
        self.n_latents = latent_dim


def test_multimodal_tristream_vae_trainer_steps():
    model = DummyMVAE()
    train_ds = DummyDataset(num_samples=3, in_dim=10, labels=False)
    val_ds = DummyDataset(num_samples=2, in_dim=10, labels=False)
    trainer_module = SignalImageTriStreamVAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=2,
        num_workers=0,
        lambda_image=1.0,
        lambda_signal=2.0,
        lr=1e-2,
        annealing_epochs=2,
        scale_factor=1e-4,
    )

    # Test forward
    dummy_signal = torch.randn(2, 10)
    dummy_image = torch.randn(2, 4, 4)
    out = trainer_module.forward(image=dummy_image, signal=dummy_signal)
    assert isinstance(out, tuple) and len(out) == 4

    # Test training_step and validation_step
    batch = (torch.ones(2, 10), torch.ones(2, 4, 4))
    train_loss = trainer_module.training_step(batch, batch_idx=0)
    assert isinstance(train_loss, torch.Tensor)
    assert torch.isfinite(train_loss)
    val_loss = trainer_module.validation_step(batch, batch_idx=0)
    assert val_loss is not None

    # Test optimizer configuration
    optim = trainer_module.configure_optimizers()
    assert isinstance(optim, torch.optim.Adam)
    assert any([p.requires_grad for p in optim.param_groups[0]["params"]])


def test_multimodal_trainer_step_and_metrics():
    torch.manual_seed(0)
    model = SignalImageFineTuningTrainer(DummyPretrainedModel(), num_classes=2, lr=1e-3)

    # Use DummyDataset with labels=True (returns image, signal, label)
    dataset = DummyDataset(num_samples=6, in_dim=5, labels=True)
    batch = (
        dataset.images[:2],
        dataset.signals[:2],
        dataset.all_labels[:2],
    )
    # Normal case (should hit success branches)
    logits = model.forward(batch[0], batch[1])
    assert logits.shape == (2, 2)

    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    model.on_train_epoch_end()

    model.validation_step(batch, batch_idx=0)
    model.validation_step(batch, batch_idx=1)
    model.on_validation_epoch_end()

    optim = model.configure_optimizers()
    assert isinstance(optim, torch.optim.Adam)
    assert any([p.requires_grad for p in optim.param_groups[0]["params"]])
    assert model.validation_step_outputs == []
    assert model.val_losses == []

    # Construct outputs so that roc_auc_score and matthews_corrcoef will fail
    model.validation_step_outputs = []
    # Create a batch with only one class present (invalid for AUC)
    labels = torch.zeros(3, dtype=torch.long)
    preds = torch.zeros(3, dtype=torch.long)
    probs = torch.zeros(3)  # Shape (3,), should be fine for code
    model.validation_step_outputs.append((labels, preds, probs))
    # MCC for one class should return 0, but to trigger except, pass empty
    model.val_losses = [torch.tensor(0.5)]
    model.on_validation_epoch_end()  # Should now hit both except branches

    model.validation_step_outputs = []
    model.validation_step_outputs.append((torch.tensor([]), torch.tensor([]), torch.tensor([])))
    model.on_validation_epoch_end()

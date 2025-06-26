import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kale.pipeline.multimodal_trainer import SignalImageFineTuningTrainer, SignalImageTriStreamVAETrainer


class DummyMVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, image=None, signal=None):
        batch_size = image.size(0) if image is not None else signal.size(0)
        recon_image = torch.zeros(batch_size, 4, 4)
        recon_signal = torch.zeros(batch_size, 10)
        mean = torch.zeros(batch_size, 6)
        log_var = torch.zeros(batch_size, 6)
        return recon_image, recon_signal, mean, log_var


class DummyDataset(Dataset):
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
            return self.signals[idx], self.images[idx]


def test_multimodal_tristream_vae_trainer_steps():
    model = DummyMVAE()
    train_ds = DummyDataset(num_samples=6, in_dim=10, labels=False)
    val_ds = DummyDataset(num_samples=3, in_dim=10, labels=False)
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

    # -- Forward, train/val step, optimizer --
    dummy_signal = torch.randn(2, 10)
    dummy_image = torch.randn(2, 4, 4)
    out = trainer_module.forward(image=dummy_image, signal=dummy_signal)
    assert isinstance(out, tuple) and len(out) == 4

    batch = (torch.ones(2, 10), torch.ones(2, 4, 4))
    train_loss = trainer_module.training_step(batch, batch_idx=0)
    assert isinstance(train_loss, torch.Tensor)
    assert torch.isfinite(train_loss)
    val_loss = trainer_module.validation_step(batch, batch_idx=0)
    assert isinstance(val_loss, torch.Tensor)

    # -- Optimizer config --
    optim = trainer_module.configure_optimizers()
    assert isinstance(optim, torch.optim.Adam)
    assert any([p.requires_grad for p in optim.param_groups[0]["params"]])

    # -- train/val dataloader with a val_dataset (tests DataLoader branch) --
    train_loader = trainer_module.train_dataloader()
    val_loader = trainer_module.val_dataloader()
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert isinstance(train_batch, (tuple, list)) and len(train_batch) == 2
    assert isinstance(val_batch, (tuple, list)) and len(val_batch) == 2
    assert train_batch[0].shape[0] == trainer_module.batch_size or train_batch[0].shape[0] == len(train_ds)
    assert val_batch[0].shape[0] <= trainer_module.batch_size

    # -- test val_dataloader when val_dataset=None (tests the else branch) --
    trainer_module_no_val = SignalImageTriStreamVAETrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=None,
        batch_size=2,
        num_workers=0,
        lambda_image=1.0,
        lambda_signal=2.0,
        lr=1e-2,
        annealing_epochs=2,
        scale_factor=1e-4,
    )
    val_loader_none = trainer_module_no_val.val_dataloader()
    assert val_loader_none is None


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


def test_multimodal_trainer_step_and_metrics():
    torch.manual_seed(0)
    # Added hidden_dim argument for flexibility in classifier
    model = SignalImageFineTuningTrainer(DummyPretrainedModel(), num_classes=2, lr=1e-3, hidden_dim=16)

    # Use DummyDataset with labels=True (returns image, signal, label)
    dataset = DummyDataset(num_samples=6, in_dim=5, labels=True)
    batch = (
        dataset.images[:2],
        dataset.signals[:2],
        dataset.all_labels[:2],
    )
    # Test forward (logits shape)
    logits = model.forward(batch[0], batch[1])
    assert logits.shape == (2, 2)

    # Test training_step (returns loss tensor)
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    model.on_train_epoch_end()  # logs average loss and clears train_losses

    # Test validation_step and metrics
    model.validation_step(batch, batch_idx=0)
    model.validation_step(batch, batch_idx=1)
    model.on_validation_epoch_end()

    optim = model.configure_optimizers()
    assert isinstance(optim, torch.optim.Adam)
    assert any([p.requires_grad for p in optim.param_groups[0]["params"]])

    # TorchMetrics resets to initial state after epoch end, so no custom lists to check
    # For edge-case robustness, check .val_losses is reset
    assert model.val_losses == []

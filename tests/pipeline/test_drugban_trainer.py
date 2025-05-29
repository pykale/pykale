import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from kale.pipeline.drugban_trainer import DrugbanTrainer
from kale.embed.ban import DrugBAN

@pytest.fixture
def dummy_config():
    return {
        "solver_lr": 1e-3,
        "num_classes": 2,
        "batch_size": 4,
        "is_da": False,
        "solver_da_lr": 1e-3,
        "da_init_epoch": 1,
        "da_method": "CDAN",
        "original_random": False,
        "use_da_entropy": True,
        "da_random_layer": True,
        "da_random_dim": 64,
        "decoder_in_dim": 128,
    }


@pytest.fixture
def dummy_model():
    model = MagicMock()
    # Forward returns: vec_drug, vec_protein, f, score
    model.return_value = (torch.rand(4, 10), torch.rand(4, 10), torch.rand(4, 10), torch.rand(4, 1))
    return model


@pytest.fixture
def dummy_batch():
    # Simulate a batch: drug, protein, labels
    return (torch.rand(4, 10), torch.rand(4, 10), torch.randint(0, 2, (4,)))


@pytest.fixture
def dummy_da_batch():
    source = (torch.rand(4, 10), torch.rand(4, 10), torch.randint(0, 2, (4,)))
    target = (torch.rand(4, 10), torch.rand(4, 10))
    return (source, target)


def test_init_no_da(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    assert trainer.model == dummy_model
    assert trainer.is_da is False


def test_init_with_da(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": True})
    assert trainer.is_da is True


def test_configure_optimizers_no_da(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    opt = trainer.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_with_da(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": True})
    opt = trainer.configure_optimizers()
    assert isinstance(opt, (list, tuple)) and all(isinstance(o, torch.optim.Optimizer) for o in opt)


def test_training_step_no_da(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    trainer.optimizers = lambda: torch.optim.SGD(trainer.parameters(), lr=1e-3)
    loss = trainer.training_step(dummy_batch, 0)
    assert loss is not None


def test_training_step_with_da(dummy_model, dummy_config, dummy_da_batch):
    dummy_config["is_da"] = True
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.optimizers = lambda: [torch.optim.SGD(trainer.parameters(), lr=1e-3),
                                  torch.optim.SGD(trainer.parameters(), lr=1e-3)]
    trainer.current_epoch = 2  # Trigger DA phase
    loss = trainer.training_step(dummy_da_batch, 0)
    assert loss is not None


def test_validation_step(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    trainer.validation_step(dummy_batch, 0)
    # Metrics and log called (no assertion)


def test_test_step(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    trainer.test_step(dummy_batch, 0)
    # Metrics and log called (no assertion)


def test_on_validation_epoch_end(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    trainer.valid_metrics = MagicMock()
    trainer.valid_metrics.compute.return_value = {"val_acc": 0.9}
    trainer.valid_metrics.reset = MagicMock()
    trainer.on_validation_epoch_end()


def test_on_test_epoch_end(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    trainer.test_metrics = MagicMock()
    trainer.test_metrics.compute.return_value = {"test_acc": 0.9}
    trainer.test_metrics.reset = MagicMock()
    trainer.on_test_epoch_end()


def test_compute_entropy_weights(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **{**dummy_config, "is_da": False})
    logits = torch.rand(4, 1)
    weights = trainer._compute_entropy_weights(logits)
    assert weights.shape == logits.shape

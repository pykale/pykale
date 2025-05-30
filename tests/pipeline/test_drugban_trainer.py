from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torchmetrics import Accuracy, AUROC, F1Score, MetricCollection, Recall, Specificity

from kale.pipeline.drugban_trainer import DrugbanTrainer


# Dummy simple model for tests
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, v_d, v_p):
        return v_d, v_p, torch.cat((v_d, v_p), dim=1), self.fc(v_d)


@pytest.fixture
def dummy_model():
    return SimpleModel()


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
        "use_da_entropy": False,
        "da_random_layer": False,
        "da_random_dim": 64,
        "decoder_in_dim": 10,
    }


@pytest.fixture
def dummy_batch():
    v_d = torch.rand(4, 10)
    v_p = torch.rand(4, 10)
    labels = torch.randint(0, 2, (4,))
    return v_d, v_p, labels


@pytest.fixture
def dummy_da_batch():
    v_d_source = torch.rand(4, 10)
    v_p_source = torch.rand(4, 10)
    labels_source = torch.randint(0, 2, (4,))
    v_d_target = torch.rand(4, 10)
    v_p_target = torch.rand(4, 10)
    return ((v_d_source, v_p_source, labels_source), (v_d_target, v_p_target))


def test_init(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    assert trainer.model == dummy_model


def test_configure_optimizers(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    opt = trainer.configure_optimizers()
    assert opt is not None


def test_training_step_no_da(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()
    loss = trainer.training_step(dummy_batch, 0)
    assert loss is not None


class DummyDiscriminator(nn.Module):
    def forward(self, x):
        return torch.ones(x.size(0), 2)


def test_training_step_with_da(dummy_model, dummy_config, dummy_da_batch):
    dummy_config["is_da"] = True
    dummy_config["da_random_layer"] = False

    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()
    trainer._trainer.current_epoch = 2

    # Mock forward to return expected outputs with feature dimensions
    dummy_model.forward = lambda v_d, v_p: (
        torch.randn(v_d.shape[0], 10),  # vec_drug
        torch.randn(v_p.shape[0], 10),  # vec_protein
        torch.randn(v_d.shape[0], 10),  # f
        torch.randn(v_d.shape[0], 2),  # score
    )

    # Correct discriminator with expected input feature dimension (20)
    class DummyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 2)  # Updated in_features to match feature size

        def forward(self, x):
            return self.fc(x)

    trainer.domain_discriminator = DummyDiscriminator()

    # Return both optimizers: one for model, one for discriminator
    trainer.optimizers = lambda: (
        torch.optim.SGD(trainer.model.parameters(), lr=1e-3),
        torch.optim.SGD(trainer.domain_discriminator.parameters(), lr=1e-3),
    )

    # Call training_step with dummy batch
    loss = trainer.training_step(dummy_da_batch, 0)
    assert loss is not None


def test_validation_step(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.log = MagicMock()
    trainer._trainer = MagicMock()
    trainer.validation_step(dummy_batch, 0)


def test_test_step(dummy_model, dummy_config, dummy_batch):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.log = MagicMock()
    trainer._trainer = MagicMock()
    trainer.test_step(dummy_batch, 0)


def test_on_validation_epoch_end(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    # Re-instantiate the metrics correctly
    if trainer.num_classes <= 2:
        metrics = MetricCollection(
            [
                AUROC("binary", average="none", num_classes=trainer.num_classes),
                F1Score("binary", average="none", num_classes=trainer.num_classes),
                Recall("binary", average="none", num_classes=trainer.num_classes),
                Specificity("binary", average="none", num_classes=trainer.num_classes),
                Accuracy("binary", average="none", num_classes=trainer.num_classes),
            ]
        )
    else:
        metrics = MetricCollection(
            [
                AUROC("multiclass", average="none", num_classes=trainer.num_classes),
                F1Score("multiclass", average="none", num_classes=trainer.num_classes),
                Recall("multiclass", average="none", num_classes=trainer.num_classes),
                Specificity("multiclass", average="none", num_classes=trainer.num_classes),
                Accuracy("multiclass", average="none", num_classes=trainer.num_classes),
            ]
        )
    trainer.valid_metrics = metrics
    trainer.valid_metrics.update = MagicMock()
    trainer.valid_metrics.compute = MagicMock(return_value={"val_acc": 0.9})
    trainer.valid_metrics.reset = MagicMock()
    trainer.log_dict = MagicMock()
    trainer.on_validation_epoch_end()


def test_on_test_epoch_end(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    if trainer.num_classes <= 2:
        metrics = MetricCollection(
            [
                AUROC("binary", average="none", num_classes=trainer.num_classes),
                F1Score("binary", average="none", num_classes=trainer.num_classes),
                Recall("binary", average="none", num_classes=trainer.num_classes),
                Specificity("binary", average="none", num_classes=trainer.num_classes),
                Accuracy("binary", average="none", num_classes=trainer.num_classes),
            ]
        )
    else:
        metrics = MetricCollection(
            [
                AUROC("multiclass", average="none", num_classes=trainer.num_classes),
                F1Score("multiclass", average="none", num_classes=trainer.num_classes),
                Recall("multiclass", average="none", num_classes=trainer.num_classes),
                Specificity("multiclass", average="none", num_classes=trainer.num_classes),
                Accuracy("multiclass", average="none", num_classes=trainer.num_classes),
            ]
        )
    trainer.test_metrics = metrics
    trainer.test_metrics.update = MagicMock()
    trainer.test_metrics.compute = MagicMock(return_value={"test_acc": 0.85})
    trainer.test_metrics.reset = MagicMock()
    trainer.log_dict = MagicMock()
    trainer.on_test_epoch_end()


def test_compute_entropy_weights(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    logits = torch.rand(4, 1)
    weights = trainer._compute_entropy_weights(logits)
    assert weights.shape == torch.Size([4])  # Adjusted assertion

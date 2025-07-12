from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from torch import nn
from torchmetrics import Accuracy, AUROC, F1Score, MetricCollection, Recall, Specificity

from kale.embed.ban import RandomLayer
from kale.pipeline.drugban_trainer import DrugbanTrainer
from kale.predict.class_domain_nets import DomainNetSmallImage


# Dummy simple model for tests
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, v_d, v_p, mode="train"):
        # Training mode returns four outputs: drug, protein, f, score
        # Evaluation mode returns five outputs: drug, protein, f, score, attention weights
        if mode == "train":
            return v_d, v_p, torch.cat((v_d, v_p), dim=1), self.fc(v_d)
        if mode == "eval":
            return v_d, v_p, torch.cat((v_d, v_p), dim=1), self.fc(v_d), self.fc(v_d)


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
    # Add enough dummy predictions and targets to avoid empty sequence errors
    trainer.test_preds.append(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    trainer.test_targets.append(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]))
    trainer.on_test_epoch_end()


def test_compute_entropy_weights(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    logits = torch.rand(4, 1)
    weights = trainer._compute_entropy_weights(logits)
    assert weights.shape == torch.Size([4])  # Adjusted assertion


def test_random_layer_with_original_random(dummy_model, dummy_config):
    dummy_config["is_da"] = True
    dummy_config["da_random_layer"] = True
    dummy_config["original_random"] = True

    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)

    # Check that random_layer is instance of RandomLayer
    assert isinstance(trainer.random_layer, RandomLayer)


def test_random_layer_disabled(dummy_model, dummy_config):
    dummy_config["is_da"] = True
    dummy_config["da_random_layer"] = False

    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)

    assert trainer.random_layer is False


def test_multiclass_metrics_and_loss(dummy_batch):
    dummy_config = {
        "solver_lr": 1e-3,
        "num_classes": 3,  # Multiclass
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

    model = SimpleModel()
    model.fc = nn.Linear(10, 3)  # Adjust to 3-class output

    trainer = DrugbanTrainer(model=model, **dummy_config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()

    # Patch metrics to bypass actual tensor shape checking in multiclass
    trainer.valid_metrics.update = MagicMock()
    trainer.test_metrics.update = MagicMock()

    trainer.log = MagicMock()

    # Call training_step with multiclass batch
    loss = trainer.training_step(dummy_batch, 0)
    assert loss is not None

    # Call validation and test steps with multiclass
    trainer.validation_step(dummy_batch, 0)
    trainer.test_step(dummy_batch, 0)


def test_training_step_invalid_da_method(dummy_model, dummy_config, dummy_da_batch):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()
    trainer._trainer.current_epoch = dummy_config["da_init_epoch"] + 1

    dummy_model.forward = lambda v_d, v_p: (
        torch.randn(v_d.shape[0], 10),
        torch.randn(v_p.shape[0], 10),
        torch.randn(v_d.shape[0], 10),
        torch.randn(v_d.shape[0], 2),
    )

    class DummyDiscriminator(nn.Module):
        def forward(self, x):
            return torch.ones(x.size(0), 2)

    trainer.domain_discriminator = DummyDiscriminator()

    trainer.optimizers = lambda: (
        torch.optim.SGD(trainer.model.parameters(), lr=1e-3),
        torch.optim.SGD(trainer.domain_discriminator.parameters(), lr=1e-3),
    )

    # Simple exception check (no regex)
    with pytest.raises(ValueError):
        trainer.training_step(dummy_da_batch, 0)


def test_on_train_epoch_start(dummy_model, dummy_config):
    trainer = DrugbanTrainer(model=dummy_model, **dummy_config)
    trainer.log = MagicMock()

    # Case 1: current_epoch < da_init_epoch
    with patch.object(DrugbanTrainer, "current_epoch", new_callable=PropertyMock) as mock_epoch:
        mock_epoch.return_value = 0
        trainer.da_init_epoch = 1
        trainer.epoch_lamb_da = 0
        trainer.on_train_epoch_start()
        trainer.log.assert_called_with(
            "DA loss lambda", trainer.epoch_lamb_da, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        assert trainer.epoch_lamb_da == 0

    # Case 2: current_epoch >= da_init_epoch
    with patch.object(DrugbanTrainer, "current_epoch", new_callable=PropertyMock) as mock_epoch:
        mock_epoch.return_value = 2
        trainer.on_train_epoch_start()
        trainer.log.assert_called_with(
            "DA loss lambda", trainer.epoch_lamb_da, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        assert trainer.epoch_lamb_da == 1


def test_domain_discriminator_and_random_layer_branches(dummy_model, dummy_config):
    # da_random_layer=True, original_random=False
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = True
    config["original_random"] = False
    trainer = DrugbanTrainer(model=dummy_model, **config)
    assert isinstance(trainer.domain_discriminator, DomainNetSmallImage)
    assert isinstance(trainer.random_layer, torch.nn.Linear)
    assert trainer.random_layer.in_features == config["decoder_in_dim"] * config["num_classes"]
    assert trainer.random_layer.out_features == config["da_random_dim"]
    # All params require_grad should be False
    for param in trainer.random_layer.parameters():
        assert not param.requires_grad

    # da_random_layer=True, original_random=True
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = True
    config["original_random"] = True
    trainer = DrugbanTrainer(model=dummy_model, **config)
    assert isinstance(trainer.domain_discriminator, DomainNetSmallImage)
    assert isinstance(trainer.random_layer, RandomLayer)

    # da_random_layer=False
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = False
    trainer = DrugbanTrainer(model=dummy_model, **config)
    assert isinstance(trainer.domain_discriminator, DomainNetSmallImage)
    assert trainer.random_layer is False


def test_training_step_with_da_entropy(dummy_model, dummy_config, dummy_da_batch):
    # Setup config for DA with entropy
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = False
    config["use_da_entropy"] = True

    trainer = DrugbanTrainer(model=dummy_model, **config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()

    # Patch current_epoch to be in DA phase
    with patch.object(DrugbanTrainer, "current_epoch", new_callable=PropertyMock) as mock_epoch:
        mock_epoch.return_value = config["da_init_epoch"]

        # Mock model forward to return correct shapes
        batch_size = config["batch_size"]
        num_classes = config["num_classes"]
        dummy_model.forward = lambda v_d, v_p: (
            torch.randn(batch_size, 10),  # vec_drug
            torch.randn(batch_size, 10),  # vec_protein
            torch.randn(batch_size, 10),  # f
            torch.randn(batch_size, num_classes),  # score
        )

        # Patch domain_discriminator to accept correct input
        class DummyDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(20, num_classes)

            def forward(self, x):
                return self.fc(x)

        trainer.domain_discriminator = DummyDiscriminator()

        # Patch optimizers
        trainer.optimizers = lambda: (
            torch.optim.SGD(trainer.model.parameters(), lr=1e-3),
            torch.optim.SGD(trainer.domain_discriminator.parameters(), lr=1e-3),
        )

        # Should run without error and return a loss
        loss = trainer.training_step(dummy_da_batch, 0)
        assert loss is not None


def test_training_step_da_target_branch_original_random(dummy_model, dummy_config, dummy_da_batch):
    # DA config with original_random True
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = True
    config["original_random"] = True

    trainer = DrugbanTrainer(model=dummy_model, **config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()

    with patch.object(DrugbanTrainer, "current_epoch", new_callable=PropertyMock) as mock_epoch:
        mock_epoch.return_value = config["da_init_epoch"]
        batch_size = config["batch_size"]
        num_classes = config["num_classes"]
        # Mock model forward to return correct shapes
        dummy_model.forward = lambda v_d, v_p: (
            torch.randn(batch_size, 10),  # vec_drug
            torch.randn(batch_size, 10),  # vec_protein
            torch.randn(batch_size, 10),  # f
            torch.randn(batch_size, num_classes),  # score
        )

        # Patch domain_discriminator to accept correct input
        class DummyDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(config["da_random_dim"], num_classes)

            def forward(self, x):
                return self.fc(x)

        trainer.domain_discriminator = DummyDiscriminator()
        # Patch optimizers to always have a parameter
        dummy_param = torch.zeros(1, requires_grad=True)
        trainer.optimizers = lambda: (
            torch.optim.SGD([dummy_param], lr=1e-3),
            torch.optim.SGD([dummy_param], lr=1e-3),
        )
        # Should run without error and return a loss
        loss = trainer.training_step(dummy_da_batch, 0)
        assert loss is not None


def test_training_step_da_target_branch_random_layer(dummy_model, dummy_config, dummy_da_batch):
    # DA config with original_random False and random_layer True
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = True
    config["original_random"] = False

    trainer = DrugbanTrainer(model=dummy_model, **config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()

    with patch.object(DrugbanTrainer, "current_epoch", new_callable=PropertyMock) as mock_epoch:
        mock_epoch.return_value = config["da_init_epoch"]
        batch_size = config["batch_size"]
        num_classes = config["num_classes"]
        # Mock model forward to return correct shapes
        dummy_model.forward = lambda v_d, v_p: (
            torch.randn(batch_size, 10),  # vec_drug
            torch.randn(batch_size, 10),  # vec_protein
            torch.randn(batch_size, 10),  # f
            torch.randn(batch_size, num_classes),  # score
        )

        # Patch domain_discriminator to accept correct input
        class DummyDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(config["da_random_dim"], num_classes)

            def forward(self, x):
                return self.fc(x)

        trainer.domain_discriminator = DummyDiscriminator()
        # Patch optimizers to always have a parameter
        dummy_param = torch.zeros(1, requires_grad=True)
        trainer.optimizers = lambda: (
            torch.optim.SGD([dummy_param], lr=1e-3),
            torch.optim.SGD([dummy_param], lr=1e-3),
        )
        # Should run without error and return a loss
        loss = trainer.training_step(dummy_da_batch, 0)
        assert loss is not None


def test_configure_optimizers_with_da(dummy_model, dummy_config):
    config = dummy_config.copy()
    config["is_da"] = True
    config["da_random_layer"] = False
    trainer = DrugbanTrainer(model=dummy_model, **config)

    # Patch domain_discriminator to have parameters
    class DummyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 2)

        def forward(self, x):
            return self.fc(x)

    trainer.domain_discriminator = DummyDiscriminator()
    opt, opt_da = trainer.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert isinstance(opt_da, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == config["solver_lr"]
    assert opt_da.param_groups[0]["lr"] == config["solver_da_lr"]


def test_training_step_binary_class():
    # Set up config for binary classification (num_classes == 1)
    dummy_config = {
        "solver_lr": 1e-3,
        "num_classes": 1,
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

    class BinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, v_d, v_p):
            return v_d, v_p, torch.cat((v_d, v_p), dim=1), self.fc(v_d)  # shape [batch, 1]

    model = BinaryModel()
    # Create a batch with correct shapes
    v_d = torch.rand(4, 10)
    v_p = torch.rand(4, 10)
    labels = torch.randint(0, 2, (4,)).float()
    dummy_batch = (v_d, v_p, labels)
    trainer = DrugbanTrainer(model=model, **dummy_config)
    trainer.manual_backward = lambda loss: loss.backward()
    trainer._trainer = MagicMock()
    loss = trainer.training_step(dummy_batch, 0)
    assert loss is not None

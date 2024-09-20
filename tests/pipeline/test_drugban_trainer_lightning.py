from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from kale.pipeline.drugban_trainer_lightning import DrugbanTrainer


# Helper function to create a dummy config
def get_dummy_config(da_use=False):
    return {
        "SOLVER": {"LR": 0.001, "DA_LR": 0.001},
        "DECODER": {"BINARY": 1, "IN_DIM": 256},
        "DA": {
            "USE": da_use,
            "METHOD": "CDAN",
            "INIT_EPOCH": 5,
            "RANDOM_LAYER": False,
            "ORIGINAL_RANDOM": False,
            "RANDOM_DIM": 128,
        },
    }


# Mock for the model forward pass
def mock_model_forward(v_d, v_p):
    batch_size = v_d.size(0)
    f = torch.randn(batch_size, 256)  # Feature size
    score = torch.randn(batch_size, 1)  # Binary classification score
    return v_d, v_p, f, score


@pytest.fixture
def dummy_trainer():
    # Create a mock model and discriminator
    model = MagicMock(spec=nn.Module)
    model.forward.side_effect = mock_model_forward

    discriminator = MagicMock(spec=nn.Module)

    # Instantiate the trainer with a mock model, discriminator, and dummy config
    config = get_dummy_config(da_use=False)
    trainer = DrugbanTrainer(model=model, discriminator=discriminator, **config)
    return trainer


def test_trainer_initialization(dummy_trainer):
    # Test that the trainer initializes correctly
    assert dummy_trainer.solver_lr == 0.001
    assert dummy_trainer.da_use is False
    assert isinstance(dummy_trainer.model, nn.Module)


def test_optimizer_configuration_without_da(dummy_trainer):
    # Test optimizer configuration without domain adaptation
    optimizers = dummy_trainer.configure_optimizers()
    assert len(optimizers) == 1  # Only one optimizer for the model


@pytest.mark.parametrize("da_use", [True, False])
def test_training_step(da_use):
    # Mock a sample batch of data
    v_d = torch.randn(32, 256)
    v_p = torch.randn(32, 256)
    labels = torch.randint(0, 2, (32,))
    train_batch = (v_d, v_p, labels)

    # Initialize a dummy trainer with or without DA
    config = get_dummy_config(da_use=da_use)
    model = MagicMock(spec=nn.Module)
    model.forward.side_effect = mock_model_forward
    discriminator = MagicMock(spec=nn.Module)
    trainer = DrugbanTrainer(model=model, discriminator=discriminator, **config)

    # Mock optimizers
    optimizer = MagicMock()
    if da_use:
        trainer.optimizers = lambda: (optimizer, optimizer)
    else:
        trainer.optimizers = lambda: optimizer

    # Run a training step
    loss = trainer.training_step(train_batch, batch_idx=0)

    # Check if the loss is computed correctly
    assert loss is not None
    optimizer.step.assert_called()  # Ensure optimizer step is called


def test_validation_step(dummy_trainer):
    # Mock a sample validation batch
    v_d = torch.randn(32, 256)
    v_p = torch.randn(32, 256)
    labels = torch.randint(0, 2, (32,))
    val_batch = (v_d, v_p, labels)

    # Run the validation step
    dummy_trainer.validation_step(val_batch, batch_idx=0)

    # Ensure validation metrics are updated
    dummy_trainer.valid_metrics.update.assert_called()


def test_optimizer_configuration_with_da():
    # Initialize a dummy trainer with DA enabled
    config = get_dummy_config(da_use=True)
    model = MagicMock(spec=nn.Module)
    discriminator = MagicMock(spec=nn.Module)
    trainer = DrugbanTrainer(model=model, discriminator=discriminator, **config)

    # Test optimizer configuration with domain adaptation
    optimizers = trainer.configure_optimizers()
    assert len(optimizers) == 2  # Two optimizers for model and discriminator

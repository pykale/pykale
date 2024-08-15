import pytest
import torch
from unittest.mock import MagicMock
from kale.pipeline.drugban_trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def model():
    model = MagicMock()
    return model
@pytest.fixture
def optimiser():
    optim = MagicMock()
    return optim
@pytest.fixture
def data():
    # Create dummy data for testing. The dimension is following the CIFAR10 dataset.
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    return TensorDataset(x, y)


@pytest.fixture
def dataloader(data):
    # Create a DataLoader for the dummy data.
    return DataLoader(data, batch_size=4)

@pytest.fixture
def config():
    return {
        "SOLVER" : {"MAX_EPOCH": 1, "BATCH_SIZE": 64},
        "DA": {"USE": False, "METHOD": "CDAN", "INIT_EPOCH": 10, "LAMB_DA": 1, "USE_ENTROPY": True, "RANDOM_LAYER": False,
               "ORIGINAL_RANDOM": False, "RANDOM_DIM": None},
        "DECODER": {"IN_DIM": 256, "BINARY": 1},
        "RESULT": {"OUTPUT_DIR": "./result", "SAVE_MODEL": True}
    }


class TestTrainer:
    @pytest.fixture
    def trainer(self, model, optimiser, config, dataloader, ):
        return Trainer(
        model = model,
        optim = optimiser,
        device=torch.device("cpu"),
        config = config,
        train_dataloader = dataloader,
        val_dataloader = dataloader,
        test_dataloader = dataloader,
        **config
        )

    def test_lambda_decay(self, trainer):
        decay_factor = trainer.da_lambda_decay()
        assert isinstance(decay_factor, float)

    def test_compute_entropy_weights(self, trainer):
        logits = torch.randn(8, 2)
        weights = trainer._compute_entropy_weights(logits)
        assert isinstance(weights, torch.Tensor)

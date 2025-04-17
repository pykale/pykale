import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from yacs.config import CfgNode as CfgNode

from kale.pipeline.drugban_trainer import Trainer


@pytest.fixture
def mock_data():
    # Create dummy data for testing.
    x1 = torch.randn(8, 10)  # Simulated feature set 1
    x2 = torch.randn(8, 10)  # Simulated feature set 2
    y = torch.randint(0, 2, (8,))  # Simulated labels
    return TensorDataset(x1, x2, y)


@pytest.fixture
def dataloaders(mock_data):
    # Create DataLoaders for train, validation, and test sets.
    train_loader = DataLoader(mock_data, batch_size=4)
    valid_loader = DataLoader(mock_data, batch_size=4)
    test_loader = DataLoader(mock_data, batch_size=4)
    return train_loader, valid_loader, test_loader


@pytest.fixture
def mock_model():
    # Create a simple mock model for testing.
    class MockModel(nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x1, x2):
            x1 = self.fc1(x1)
            x2 = self.fc1(x2)
            return x1, x2, torch.cat((x1, x2), dim=1), self.fc2(x1 + x2)

    return MockModel()


@pytest.fixture
def testing_cfg():
    _C = CfgNode()

    _C.DECODER = CfgNode()
    _C.DECODER.NAME = "MLP"
    _C.DECODER.IN_DIM = 256
    _C.DECODER.HIDDEN_DIM = 512
    _C.DECODER.OUT_DIM = 128
    _C.DECODER.BINARY = 1

    _C.SOLVER = CfgNode()
    _C.SOLVER.MAX_EPOCH = 2
    _C.SOLVER.BATCH_SIZE = 4
    _C.SOLVER.NUM_WORKERS = 0
    _C.SOLVER.LEARNING_RATE = 5e-5
    _C.SOLVER.DA_LEARNING_RATE = 1e-3
    _C.SOLVER.SEED = 2048

    _C.RESULT = CfgNode()
    _C.RESULT.OUTPUT_DIR = "./result"
    _C.RESULT.SAVE_MODEL = False

    _C.DA = CfgNode()
    _C.DA.TASK = False
    _C.DA.METHOD = "CDAN"
    _C.DA.USE = False
    _C.DA.INIT_EPOCH = 1
    _C.DA.LAMB_DA = 1
    _C.DA.RANDOM_LAYER = False
    _C.DA.ORIGINAL_RANDOM = False
    _C.DA.RANDOM_DIM = None
    _C.DA.USE_ENTROPY = False

    return _C.clone()


@pytest.fixture
def testing_DA_cfg():
    _C = CfgNode()

    _C.DECODER = CfgNode()
    _C.DECODER.NAME = "MLP"
    _C.DECODER.IN_DIM = 256
    _C.DECODER.HIDDEN_DIM = 512
    _C.DECODER.OUT_DIM = 128
    _C.DECODER.BINARY = 1

    _C.SOLVER = CfgNode()
    _C.SOLVER.MAX_EPOCH = 2
    _C.SOLVER.BATCH_SIZE = 4
    _C.SOLVER.NUM_WORKERS = 0
    _C.SOLVER.LEARNING_RATE = 5e-5
    _C.SOLVER.DA_LEARNING_RATE = 1e-3
    _C.SOLVER.SEED = 2048

    _C.RESULT = CfgNode()
    _C.RESULT.OUTPUT_DIR = "./result"
    _C.RESULT.SAVE_MODEL = False

    _C.DA = CfgNode()
    _C.DA.TASK = True
    _C.DA.METHOD = "CDAN"
    _C.DA.USE = True
    _C.DA.INIT_EPOCH = 1
    _C.DA.LAMB_DA = 1
    _C.DA.RANDOM_LAYER = False
    _C.DA.ORIGINAL_RANDOM = False
    _C.DA.RANDOM_DIM = 128
    _C.DA.USE_ENTROPY = False

    return _C.clone()


@pytest.fixture
def trainer(mock_model, dataloaders, testing_cfg):
    train_loader, valid_loader, test_loader = dataloaders
    return Trainer(
        model=mock_model,
        device=torch.device("cpu"),
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        test_dataloader=test_loader,
        experiment=None,
        alpha=1.0,
        config=testing_cfg,
        **testing_cfg,
    )


@pytest.fixture
def da_trainer(mock_model, dataloaders, testing_DA_cfg):
    train_loader, valid_loader, test_loader = dataloaders
    return Trainer(
        model=mock_model,
        device=torch.device("cpu"),
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        test_dataloader=test_loader,
        experiment=None,
        alpha=1.0,
        config=testing_DA_cfg,
        **testing_DA_cfg,
    )


class TestTrainer:
    def test_train_epoch(self, trainer):
        train_loss = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_test_valid(self, trainer):
        auroc, auprc, test_loss = trainer.test(dataloader="valid")
        assert isinstance(auroc, float)
        assert isinstance(auprc, float)
        assert isinstance(test_loss, float)

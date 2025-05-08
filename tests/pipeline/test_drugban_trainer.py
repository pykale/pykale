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
    val_loader = DataLoader(mock_data, batch_size=4)
    test_loader = DataLoader(mock_data, batch_size=4)
    return train_loader, val_loader, test_loader


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
def optimiser(mock_model):
    # Create a simple optimiser for testing
    return torch.optim.Adam(mock_model.parameters(), lr=0.01)


@pytest.fixture
def testing_cfg():
    _C = CfgNode()

    # ---------------------------------------------------------------------------- #
    # MLP decoder
    # ---------------------------------------------------------------------------- #
    _C.DECODER = CfgNode()
    _C.DECODER.NAME = "MLP"
    _C.DECODER.IN_DIM = 256
    _C.DECODER.HIDDEN_DIM = 512
    _C.DECODER.OUT_DIM = 128
    _C.DECODER.BINARY = 1

    # ---------------------------------------------------------------------------- #
    # SOLVER
    # ---------------------------------------------------------------------------- #
    _C.SOLVER = CfgNode()
    _C.SOLVER.MAX_EPOCH = 2
    _C.SOLVER.BATCH_SIZE = 64
    _C.SOLVER.NUM_WORKERS = 0
    _C.SOLVER.LR = 5e-5
    _C.SOLVER.DA_LR = 1e-3
    _C.SOLVER.SEED = 2048

    # ---------------------------------------------------------------------------- #
    # RESULT
    # ---------------------------------------------------------------------------- #
    _C.RESULT = CfgNode()
    _C.RESULT.OUTPUT_DIR = "./result"
    _C.RESULT.SAVE_MODEL = True

    # ---------------------------------------------------------------------------- #
    # Domain adaptation
    # ---------------------------------------------------------------------------- #
    _C.DA = CfgNode()
    _C.DA.TASK = False  # False: 'in-domain' splitting strategy, True: 'cross-domain' splitting strategy
    _C.DA.METHOD = "CDAN"
    _C.DA.USE = False  # False: no domain adaptation, True: domain adaptation
    _C.DA.INIT_EPOCH = 10
    _C.DA.LAMB_DA = 1
    _C.DA.RANDOM_LAYER = False
    _C.DA.ORIGINAL_RANDOM = False
    _C.DA.RANDOM_DIM = None
    _C.DA.USE_ENTROPY = True

    yield _C.clone()


@pytest.fixture
def testing_DA_cfg():
    _C = CfgNode()

    # ---------------------------------------------------------------------------- #
    # MLP decoder
    # ---------------------------------------------------------------------------- #
    _C.DECODER = CfgNode()
    _C.DECODER.NAME = "MLP"
    _C.DECODER.IN_DIM = 256
    _C.DECODER.HIDDEN_DIM = 512
    _C.DECODER.OUT_DIM = 128
    _C.DECODER.BINARY = 1

    # ---------------------------------------------------------------------------- #
    # SOLVER
    # ---------------------------------------------------------------------------- #
    _C.SOLVER = CfgNode()
    _C.SOLVER.MAX_EPOCH = 2
    _C.SOLVER.BATCH_SIZE = 64
    _C.SOLVER.NUM_WORKERS = 0
    _C.SOLVER.LR = 5e-5
    _C.SOLVER.DA_LR = 1e-3
    _C.SOLVER.SEED = 2048

    # ---------------------------------------------------------------------------- #
    # RESULT
    # ---------------------------------------------------------------------------- #
    _C.RESULT = CfgNode()
    _C.RESULT.OUTPUT_DIR = "./result"
    _C.RESULT.SAVE_MODEL = True

    # ---------------------------------------------------------------------------- #
    # Domain adaptation
    # ---------------------------------------------------------------------------- #
    _C.DA = CfgNode()
    _C.DA.TASK = True  # False: 'in-domain' splitting strategy, True: 'cross-domain' splitting strategy
    _C.DA.METHOD = "CDAN"
    _C.DA.USE = True  # False: no domain adaptation, True: domain adaptation
    _C.DA.INIT_EPOCH = 1
    _C.DA.LAMB_DA = 1
    _C.DA.RANDOM_LAYER = True
    _C.DA.ORIGINAL_RANDOM = True
    _C.DA.RANDOM_DIM = 256
    _C.DA.USE_ENTROPY = False

    yield _C.clone()


@pytest.fixture
def trainer(mock_model, optimiser, dataloaders, testing_cfg):
    # Create a Trainer instance for testing.
    train_loader, val_loader, test_loader = dataloaders
    return Trainer(
        model=mock_model,
        optim=optimiser,
        device=torch.device("cpu"),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        **testing_cfg,
    )


class TestTrainer:
    def test_train_epoch(self, trainer):
        # Test a single epoch of training.
        train_loss = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_test(self, trainer):
        # Test the model evaluation function.
        auroc, auprc, test_loss = trainer.test(dataloader="val")
        assert isinstance(auroc, float)
        assert isinstance(auprc, float)
        assert isinstance(test_loss, float)

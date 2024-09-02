import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
def trainer(mock_model, dataloaders):
    # Create a Trainer instance for testing.
    train_loader, val_loader, test_loader = dataloaders
    config = {
        "SOLVER": {"MAX_EPOCH": 2, "BATCH_SIZE": 4},
        "DA": {"USE": False, "METHOD": "CDAN", "RANDOM_LAYER": False, "ORIGINAL_RANDOM": False, "INIT_EPOCH": 1, "LAMB_DA": 0.1},
        "DECODER": {"BINARY": 1, "IN_DIM": 5},
        "RESULT": {"OUTPUT_DIR": "./", "SAVE_MODEL": False},
    }
    optim = torch.optim.SGD(mock_model.parameters(), lr=0.01)
    return Trainer(
        model=mock_model,
        optim=optim,
        device=torch.device("cpu"),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=config,
    )


class TestTrainer:
    def test_train_epoch(self, trainer):
        # Test a single epoch of training.
        train_loss = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_train_da_epoch(self, trainer):
        # Test a single epoch of training with domain adaptation.
        trainer.is_da = True
        trainer.current_epoch = 2
        total_loss, model_loss, da_loss, lamb_da = trainer.train_da_epoch()
        assert isinstance(total_loss, float)
        assert isinstance(model_loss, float)
        assert isinstance(da_loss, float)
        assert lamb_da >= 0

    def test_test(self, trainer):
        # Test the model evaluation function.
        auroc, auprc, test_loss = trainer.test(dataloader="val")
        assert isinstance(auroc, float)
        assert isinstance(auprc, float)
        assert isinstance(test_loss, float)

    def test_save_result(self, trainer, tmpdir):
        # Test the saving of results.
        trainer.output_dir = tmpdir.mkdir("output")
        trainer.save_result()
        # Check if the result files are created.
        assert len(list(trainer.output_dir.iterdir())) > 0


class TestTrainerWithDA:
    @pytest.fixture
    def trainer_with_da(self, mock_model, dataloaders):
        # Create a Trainer instance with Domain Adaptation enabled.
        train_loader, val_loader, test_loader = dataloaders
        config = {
            "SOLVER": {"MAX_EPOCH": 2, "BATCH_SIZE": 4},
            "DA": {"USE": True, "METHOD": "CDAN", "RANDOM_LAYER": False, "ORIGINAL_RANDOM": False, "INIT_EPOCH": 1, "LAMB_DA": 0.1},
            "DECODER": {"BINARY": 1, "IN_DIM": 5},
            "RESULT": {"OUTPUT_DIR": "./", "SAVE_MODEL": False},
        }
        optim = torch.optim.SGD(mock_model.parameters(), lr=0.01)
        opt_da = torch.optim.SGD(mock_model.parameters(), lr=0.01)
        return Trainer(
            model=mock_model,
            optim=optim,
            device=torch.device("cpu"),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            opt_da=opt_da,
            config=config,
        )

    def test_train_da_epoch(self, trainer_with_da):
        # Test training epoch with domain adaptation.
        trainer_with_da.current_epoch = 2
        total_loss, model_loss, da_loss, lamb_da = trainer_with_da.train_da_epoch()
        assert isinstance(total_loss, float)
        assert isinstance(model_loss, float)
        assert isinstance(da_loss, float)
        assert lamb_da >= 0

    def test_compute_entropy_weights(self, trainer_with_da):
        # Test entropy weight computation for domain adaptation.
        logits = torch.randn(4, 2)
        entropy_weights = trainer_with_da._compute_entropy_weights(logits)
        assert isinstance(entropy_weights, torch.Tensor)
        assert entropy_weights.size(0) == logits.size(0)

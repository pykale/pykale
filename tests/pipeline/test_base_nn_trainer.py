import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kale.embed.image_cnn import SimpleCNNBuilder
from kale.pipeline.base_nn_trainer import BaseNNTrainer, CNNTransformerTrainer, MultimodalNNTrainer
from kale.predict.class_domain_nets import ClassNet


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
def batch(dataloader):
    # Create a batch from the DataLoader.
    return next(iter(dataloader))


class TestBaseTrainer:
    @pytest.fixture
    def trainer(self):
        # Create a default BaseNNTrainer for testing.
        return BaseNNTrainer(None, 5)

    def test_forward(self, trainer, batch):
        # Test forward pass. Return NotImplementError.
        with pytest.raises(NotImplementedError) as excinfo:
            trainer.forward(batch[0])
            assert "Forward pass needs to be defined." in str(excinfo.value)

    def test_compute_loss(self, trainer, batch):
        # Test loss computation. Return NotImplementError.
        with pytest.raises(NotImplementedError) as excinfo:
            trainer.compute_loss(batch)
            assert "Loss function needs to be defined." in str(excinfo.value)

    def test_configure_optimizers_with_default(self, trainer):
        # Test default optimizer configuration. Return a Adam optimizer.
        trainer.optimizers = ClassNet()
        # trainer._parameters = ClassNet()._parameters
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 1
        assert isinstance(optimizers, list)
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert optimizers[0].defaults["lr"] == 0.001

    def test_configure_optimizers_with_adam(self, trainer):
        # Test Adam optimizer configuration. Return a configured Adam optimizer.
        trainer.optimizers = ClassNet()
        trainer._optimizer_params = {"type": "Adam", "optim_params": {"eps": 0.2, "weight_decay": 0.3}}
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 1
        assert isinstance(optimizers, list)
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert optimizers[0].defaults["eps"] == 0.2
        assert optimizers[0].defaults["weight_decay"] == 0.3

    def test_configure_optimizers_with_sgd(self, trainer):
        # Test SGD optimizer configuration. Return a configured SGD optimizer.
        trainer.optimizers = ClassNet()
        trainer._optimizer_params = {"type": "SGD", "optim_params": {"momentum": 0.2, "weight_decay": 0.3}}
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 1
        assert isinstance(optimizers, list)
        assert isinstance(optimizers[0], torch.optim.SGD)
        assert optimizers[0].defaults["momentum"] == 0.2
        assert optimizers[0].defaults["weight_decay"] == 0.3

    def test_configure_optimizers_with_sgd_cosine_annealing(self, trainer):
        # Test SGD optimizer configuration with CosineAnnealingLR scheduler. Return a configured SGD optimizer.
        trainer.optimizers = ClassNet()
        trainer._adapt_lr = True
        trainer._optimizer_params = {"type": "SGD", "optim_params": {"momentum": 0.2, "weight_decay": 0.3}}
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 2
        assert isinstance(optimizers, tuple)
        assert isinstance(optimizers[0], list)
        assert isinstance(optimizers[1], list)
        assert isinstance(optimizers[0][0], torch.optim.SGD)
        assert isinstance(optimizers[1][0], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_configure_optimizers_with_unknown_type(self, trainer):
        # Test unknown optimizer configuration. Return NotImplementedError.
        trainer._optimizer_params = {"type": "Unknown", "optim_params": {}}
        with pytest.raises(NotImplementedError) as excinfo:
            trainer.configure_optimizers()
            assert "Unknown optimizer type Unknown." in str(excinfo.value)


class TestCNNTransformerTrainer:
    @pytest.fixture
    def trainer(self):
        # Create a CNNTransformerTrainer for testing.
        cnn = SimpleCNNBuilder([[4, 3], [16, 1]])
        classifier = ClassNet()
        return CNNTransformerTrainer(
            feature_extractor=cnn,
            task_classifier=classifier,
            lr_milestones=[1, 2],
            lr_gamma=0.1,
            optimizer={"type": "SGD", "optim_params": {"momentum": 0.2, "weight_decay": 0.3}},
            max_epochs=5,
            adapt_lr=True,
        )

    def test_forward(self, trainer, batch):
        # Test forward pass. Return torch.Tensor.
        output = trainer.forward(batch[0])
        assert isinstance(output, torch.Tensor)

    def test_compute_loss(self, trainer, batch):
        # Test loss computation. Return torch.Tensor and dict.
        loss, log_metrics = trainer.compute_loss(batch)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(log_metrics, dict)

    def test_configure_optimizers_with_default(self, trainer):
        # Test default SGD optimizer configuration. Return a configured SGD optimizer.
        trainer._adapt_lr = False
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 1
        assert isinstance(optimizers, list)
        assert isinstance(optimizers[0], torch.optim.SGD)
        assert optimizers[0].defaults["lr"] == 0.001
        assert optimizers[0].defaults["momentum"] == 0.2
        assert optimizers[0].defaults["weight_decay"] == 0.3

    def test_configure_optimizers_with_adapt_lr(self, trainer):
        # Test SGD optimizer configuration with MultiStepLR scheduler. Return a configured SGD optimizer.
        trainer._adapt_lr = True
        optimizers = trainer.configure_optimizers()
        assert len(optimizers) == 2
        assert isinstance(optimizers, tuple)
        assert isinstance(optimizers[0], list)
        assert isinstance(optimizers[1], list)
        assert isinstance(optimizers[0][0], torch.optim.SGD)
        assert isinstance(optimizers[1][0], torch.optim.lr_scheduler.MultiStepLR)

    def test_training_step(self, trainer, batch):
        # Test training step. Return torch.Tensor.
        loss = trainer.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_validation_step(self, trainer, batch):
        # Test validation step.
        trainer.validation_step(batch, batch_idx=0)

    def test_test_step(self, trainer, batch):
        # Test testing step.
        trainer.test_step(batch, batch_idx=0)


class _TestEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_TestEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class _TestFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_TestFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_concat = torch.cat(x, dim=1)
        return self.fc(x_concat)


class _TestClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(_TestClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def multimodal_model():
    encoders = [_TestEncoder(5, 10) for _ in range(2)]
    fusion = _TestFusion(20, 10)
    head = _TestClassifier(10, 2)
    model = MultimodalNNTrainer(encoders, fusion, head)
    return model


def test_compute_loss(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    loss, metrics = multimodal_model.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert set(metrics.keys()) == {"loss", "accuracy"}
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)


def test_training_step(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    loss = multimodal_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)


def test_validation_step(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    multimodal_model.validation_step(batch, 0)


def test_test_step(multimodal_model):
    x = [torch.rand(1, 5) for _ in range(2)]
    y = torch.tensor([1])
    batch = [*x, y]
    multimodal_model.test_step(batch, 0)


def test_configure_optimizers(multimodal_model):
    optimizer = multimodal_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

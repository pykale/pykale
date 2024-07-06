import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from yacs.config import CfgNode as CfgNode

from kale.embed.image_cnn import ResNet18Feature
from kale.pipeline.fewshot_trainer import ProtoNetTrainer

modes = ["train", "val", "test"]


@pytest.fixture(scope="module")
def testing_cfg():
    _C = CfgNode()
    _C.SEED = 1397
    _C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _C.MODEL = CfgNode()
    _C.MODEL.BACKBONE = "ResNet18Feature"
    _C.MODEL.PRETRAIN_WEIGHTS = None

    _C.TRAIN = CfgNode()
    _C.TRAIN.EPOCHS = 1
    _C.TRAIN.OPTIMIZER = "SGD"
    _C.TRAIN.LEARNING_RATE = 1e-3
    _C.TRAIN.NUM_CLASSES = 5
    _C.TRAIN.NUM_SUPPORT_SAMPLES = 5
    _C.TRAIN.NUM_QUERY_SAMPLES = 15

    _C.VAL = CfgNode()
    _C.VAL.NUM_CLASSES = 5
    _C.VAL.NUM_SUPPORT_SAMPLES = 5
    _C.VAL.NUM_QUERY_SAMPLES = 15
    yield _C.clone()


@pytest.fixture
def data():
    # Create dummy data for testing. The dimension is following the CIFAR10 dataset.
    x = torch.randn(5, 20, 3, 84, 84)
    y = torch.randint(0, 10, (5,))
    return TensorDataset(x, y)


@pytest.fixture
def dataloader(data):
    # Create a DataLoader for the dummy data.
    return DataLoader(data, batch_size=5)


@pytest.mark.parametrize("mode", modes)
def test_fewshot_trainer(mode, testing_cfg, dataloader):
    cfg = testing_cfg

    assert len(dataloader) > 0
    net = ResNet18Feature(weights=cfg.MODEL.PRETRAIN_WEIGHTS).to(cfg.DEVICE)
    model = ProtoNetTrainer(
        net=net,
        train_num_classes=cfg.TRAIN.NUM_CLASSES,
        train_num_support_samples=cfg.TRAIN.NUM_SUPPORT_SAMPLES,
        train_num_query_samples=cfg.TRAIN.NUM_QUERY_SAMPLES,
        val_num_classes=cfg.VAL.NUM_CLASSES,
        val_num_support_samples=cfg.VAL.NUM_SUPPORT_SAMPLES,
        val_num_query_samples=cfg.VAL.NUM_QUERY_SAMPLES,
        devices=cfg.DEVICE,
        optimizer=cfg.TRAIN.OPTIMIZER,
        lr=cfg.TRAIN.LEARNING_RATE,
    )
    trainer = pl.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=cfg.DEVICE)

    if mode == "train":
        for images, _ in dataloader:
            feature_support, feature_query = model.forward(images, cfg.TRAIN.NUM_SUPPORT_SAMPLES, cfg.TRAIN.NUM_CLASSES)
            loss, metrics = model.compute_loss(feature_support, feature_query, mode="train")
            assert isinstance(feature_query, torch.Tensor)
            assert isinstance(feature_support, torch.Tensor)
            assert isinstance(loss, torch.Tensor)
            assert isinstance(metrics, dict)
            break

        assert isinstance(model.configure_optimizers(), torch.optim.Optimizer)
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    else:
        trainer.test(model, dataloaders=dataloader)

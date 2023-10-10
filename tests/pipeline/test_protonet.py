import os
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from yacs.config import CfgNode as CfgNode

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.few_shot import NWayKShotDataset
from kale.pipeline.protonet import ProtoNetTrainer
from kale.utils.download import download_file_by_url

url = "https://github.com/pykale/data/raw/main/images/omniglot/omniglot_demo.zip"
modes = ["train", "val", "test"]

@pytest.fixture(scope="module")
def testing_cfg_model():
    _C = CfgNode()
    _C.SEED = 1397
    _C.DEVICE = "cpu"

    _C.MODEL = CfgNode()
    _C.MODEL.BACKBONE = "ResNet18Feature"
    _C.MODEL.PRETRAIN_WEIGHTS = None

    _C.TRAIN = CfgNode()
    _C.TRAIN.EPOCHS = 1
    _C.TRAIN.OPTIMIZER = "SGD"
    _C.TRAIN.LEARNING_RATE = 1e-3
    _C.TRAIN.N_WAYS = 5
    _C.TRAIN.K_SHOTS = 5
    _C.TRAIN.K_QUERIES = 15

    _C.VAL = CfgNode()
    _C.VAL.N_WAYS = 5
    _C.VAL.K_SHOTS = 5
    _C.VAL.K_QUERIES = 15
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
def test_protonet(mode, testing_cfg_model, dataloader):

    cfg_model = testing_cfg_model

    assert len(dataloader) > 0
    net = ResNet18Feature(weights=cfg_model.MODEL.PRETRAIN_WEIGHTS).to(cfg_model.DEVICE)
    model = ProtoNetTrainer(cfg=cfg_model, net=net)
    trainer = pl.Trainer(max_epochs=cfg_model.TRAIN.EPOCHS, accelerator=cfg_model.DEVICE)

    if mode == "train":
        for images, _ in dataloader:
            feature_sup, feature_que = model.forward(images, cfg_model.TRAIN.K_SHOTS, cfg_model.TRAIN.N_WAYS)
            loss, metrics = model.compute_loss(feature_sup, feature_que, mode="train")
            assert isinstance(feature_que, torch.Tensor)
            assert isinstance(feature_sup, torch.Tensor)
            assert isinstance(loss, torch.Tensor)
            assert isinstance(metrics, dict)
            break

        assert isinstance(model.configure_optimizers(), torch.optim.Optimizer)
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    else:
        trainer.test(model, dataloaders=dataloader)

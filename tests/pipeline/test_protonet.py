import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from yacs.config import CfgNode as CfgNode

from kale.embed.image_cnn import ResNet18Feature
from kale.pipeline.protonet import ProtoNetTrainer

url = "https://github.com/pykale/data/raw/main/images/omniglot/omniglot_demo.zip"
modes = ["train", "val", "test"]


@pytest.fixture(scope="module")
def testing_cfg():
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
def test_protonet(mode, testing_cfg, dataloader):
    cfg = testing_cfg

    assert len(dataloader) > 0
    net = ResNet18Feature(weights=cfg.MODEL.PRETRAIN_WEIGHTS).to(cfg.DEVICE)
    model = ProtoNetTrainer(
        net=net,
        train_n_way=cfg.TRAIN.N_WAYS,
        train_k_shot=cfg.TRAIN.K_SHOTS,
        train_k_query=cfg.TRAIN.K_QUERIES,
        val_n_way=cfg.VAL.N_WAYS,
        val_k_shot=cfg.VAL.K_SHOTS,
        val_k_query=cfg.VAL.K_QUERIES,
        devices=cfg.DEVICE,
        optimizer=cfg.TRAIN.OPTIMIZER,
        lr=cfg.TRAIN.LEARNING_RATE,
    )
    trainer = pl.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator=cfg.DEVICE)

    if mode == "train":
        for images, _ in dataloader:
            feature_sup, feature_que = model.forward(images, cfg.TRAIN.K_SHOTS, cfg.TRAIN.N_WAYS)
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

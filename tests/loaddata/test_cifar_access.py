# import glob
# import os

import pytest
import torch
from yacs.config import CfgNode as CN

from kale.loaddata.cifar_access import get_cifar

DATASET_NAMES = ["CIFAR10", "CIFAR100"]


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.SOLVER = CN()
    cfg.DATASET.ROOT = download_path
    # cfg.DATASET.DOWNLOAD = True
    cfg.SOLVER.TRAIN_BATCH_SIZE = 16
    cfg.SOLVER.TEST_BATCH_SIZE = 20
    cfg.DATASET.NUM_WORKERS = 1
    yield cfg

    # Teardown: remove data files, or not?
    # files = glob.glob(cfg.DATASET.ROOT)
    # for f in files:
    #     os.remove(f)


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_get_cifar(dataset, testing_cfg):
    cfg = testing_cfg
    cfg.DATASET.NAME = dataset
    train_loader, val_loader = get_cifar(cfg)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

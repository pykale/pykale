"""CIFAR10 or CIFAR100 dataset loading
Reference: https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/dataset.py
"""
import logging

import torch
import torchvision

from kale.prepdata.image_transform import get_transform


def get_cifar(cfg):
    """Gets training and validation data loaders for the CIFAR datasets

    Args:
        cfg: A YACS config object.
    """
    logging.info("==> Preparing to load data " + cfg.DATASET.NAME + " at " + cfg.DATASET.ROOT)
    cifar_train_transform = get_transform("cifar", augment=True)
    cifar_test_transform = get_transform("cifar", augment=False)

    if cfg.DATASET.NAME == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(
            cfg.DATASET.ROOT, train=True, download=True, transform=cifar_train_transform
        )
        val_set = torchvision.datasets.CIFAR10(
            cfg.DATASET.ROOT, train=False, download=True, transform=cifar_test_transform
        )
    elif cfg.DATASET.NAME == "CIFAR100":
        train_set = torchvision.datasets.CIFAR100(
            cfg.DATASET.ROOT, train=True, download=True, transform=cifar_train_transform
        )
        val_set = torchvision.datasets.CIFAR100(
            cfg.DATASET.ROOT, train=False, download=True, transform=cifar_test_transform
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader

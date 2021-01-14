"""CIFAR10 or CIFAR100 dataset loading

Reference: https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/dataset.py 
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from kale.prepdata.image_transform import get_transform


def get_cifar(cfg):
    """Gets training and validation data loaders for the CIFAR datasets

    Args:
        cfg: A YACS config object.
    """
    print("==> Preparing to load data " + cfg.DATASET.NAME + " at " + cfg.DATASET.ROOT)
    cifar_train_transform = get_transform("cifar", augment=True)
    cifar_test_transform = get_transform("cifar", augment=False)
    # transform = {
    #     'cifar_train': transforms.Compose([
    #         # Data augmentation
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]),
    #     'cifar_test': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    # }
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

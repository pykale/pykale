# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/dataset.py
# Under the MIT License
import os
import torch
import torchvision
import torchvision.transforms as transforms
from config import get_cfg_defaults

def construct_dataset(cfg):
    """
    args:
        cfg - a YACS config object.
    """
    print('==> Preparing data ' + cfg.DATASET.NAME + ' at ' + cfg.DATASET.ROOT)
    transform = {
        'cifar_train': transforms.Compose([
            # Data augmentation
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'cifar_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    }
    if cfg.DATASET.NAME == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(cfg.DATASET.ROOT, train=True, download=True,
                                                transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR10(cfg.DATASET.ROOT, train=False, download=True,
                                                transform=transform['cifar_test'])
    elif cfg.DATASET.NAME == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(cfg.DATASET.ROOT, train=True, download=True,
                                                transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR100(cfg.DATASET.ROOT, train=False, download=True,
                                                transform=transform['cifar_test'])
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
                                               shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
                                             shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS
                                             , pin_memory=True)

    return train_loader, val_loader

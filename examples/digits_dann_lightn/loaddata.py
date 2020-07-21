# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/dataset.py 
# Under the MIT License
import os
import torch
import torchvision
import torchvision.transforms as transforms
from config import C



def construct_dataset(): 
    print('==> Preparing data ' + C.DATASET.NAME + ' at ' + C.DATASET.ROOT)
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
    if C.DATASET.NAME == 'CIFAR10':
        train_set = torchvision.datasets.CIFAR10(C.DATASET.ROOT, train=True, download=True, 
                                                transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR10(C.DATASET.ROOT, train=False, download=True, 
                                                transform=transform['cifar_test'])
    elif C.DATASET.NAME == 'CIFAR100':
        train_set = torchvision.datasets.CIFAR100(C.DATASET.ROOT, train=True, download=True, 
                                                transform=transform['cifar_train'])
        val_set = torchvision.datasets.CIFAR100(C.DATASET.ROOT, train=False, download=True, 
                                                transform=transform['cifar_test'])
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=C.SOLVER.TRAIN_BATCH_SIZE,
                                               shuffle=True, num_workers=C.DATASET.NUM_WORKERS,
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=C.SOLVER.TEST_BATCH_SIZE,
                                             shuffle=False, num_workers=C.DATASET.NUM_WORKERS
                                             , pin_memory=True)

    return train_loader, val_loader
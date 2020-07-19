# Created by Haiping Lu from modifying https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
import os
import argparse
import warnings
import sys
# No need if pykale is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import os.path as osp
import numpy as np
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
import random
import math


import kale
import torch
from torchsummary import summary

from  config import C
import loaddata as du
# import optim as ou
import kale.utils.logger as lu
import kale.utils.seed as seed
import kale.predict.isonet as isonet
from trainer import Trainer


# Inherite and override
class CifarIsoNet(isonet.ISONet):
    
    def __init__(self):
        super(CifarIsoNet, self).__init__()
        # define network structures (override)
        self._construct()        
        # initialization
        self._network_init(C.ISON.DIRAC_INIT)

    def _construct(self):
        assert (C.ISON.DEPTH - 2) % 6 == 0, \
            'Model depth should be of the format 6n + 2 for cifar'  # Seems because this is a ResNet
        # Each stage has the same number of blocks for cifar
        d = int((C.ISON.DEPTH - 2) / 6)
        # Stem: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.stem = isonet.ResStem(w_in=3, w_out=16, has_bn=C.ISON.HAS_BN, use_srelu=C.ISON.SReLU, 
                                  kernelsize=3, stride=1, padding=1)
        # Stage 1: (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.s1 = isonet.ResStage(w_in=16, w_out=16, stride=1, transfun=C.ISON.TRANS_FUN, 
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Stage 2: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s2 = isonet.ResStage(w_in=16, w_out=32, stride=2, transfun=C.ISON.TRANS_FUN, 
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Stage 3: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s3 = isonet.ResStage(w_in=32, w_out=64, stride=2, transfun=C.ISON.TRANS_FUN,
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = isonet.ResHead(w_in=64, nc=C.DATASET.NUM_CLASSES, use_dropout=C.ISON.DROPOUT,
                                dropout_rate=C.ISON.DROPOUT_RATE)

def arg_parse():
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network on Office31')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='default', help='folder to save output', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    # ---- setup device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using device ' + device)

    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.freeze()
    seed.set_seed(C.SOLVER.SEED)
    # ---- setup logger and output ----
    output_dir = os.path.join(C.OUTPUT_DIR, C.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = lu.construct_logger('cdan', output_dir)
    logger.info('Using ' + device)
    logger.info(C.dump())    
    # ---- setup dataset ----
    train_loader, val_loader = du.construct_dataset()

    print('==> Building model..')
    net = CifarIsoNet()
    # print(net)
    net = net.to(device)
    # summary(net, (3, 32, 32))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        # Set to true will be faster but results will vary a bit (not 100% reproducible)
        # torch.backends.cudnn.benchmark = True 

    # optim = ou.construct_optim(net)
    optim = torch.optim.SGD(net.parameters(), lr=C.SOLVER.BASE_LR, momentum=C.SOLVER.MOMENTUM,
                    weight_decay=C.SOLVER.WEIGHT_DECAY, dampening=C.SOLVER.DAMPENING,
                    nesterov=C.SOLVER.NESTEROV)

    trainer = Trainer(
        device,
        train_loader,
        val_loader,
        net,
        optim,
        logger,
        output_dir,
    )

    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp['net'])
        trainer.optim.load_state_dict(cp['optim'])
        trainer.epochs = cp['epoch']
        trainer.train_acc = cp['train_accuracy']
        trainer.val_acc = cp['test_accuracy']

    trainer.train()


if __name__ == '__main__':
    main()

# This file is modified by Haiping Lu from https://github.com/HaozhiQi/ISONet/blob/master/isonet/models/isonet.py
# This file is modified from https://github.com/facebookresearch/pycls/blob/master/pycls/models/resnet.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import kale.predict.isonn as isonn

from config import C

class ISONet(nn.Module):
    """ResNet model."""

    def __init__(self):
        super(ISONet, self).__init__()
        # define network structures
        if 'CIFAR' in C.DATASET.NAME:
            self._construct_cifar()
        else:
            raise NotImplementedError
        # initialization
        self._network_init()

    def _construct_cifar(self):
        assert (C.ISON.DEPTH - 2) % 6 == 0, \
            'Model depth should be of the format 6n + 2 for cifar'  # Seems because this is a ResNet
        # Each stage has the same number of blocks for cifar
        d = int((C.ISON.DEPTH - 2) / 6)
        # Stem: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.stem = isonn.ResStem(w_in=3, w_out=16, has_bn=C.ISON.HAS_BN, use_srelu=C.ISON.SReLU, 
                                  kernelsize=3, stride=1, padding=1)
        # Stage 1: (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.s1 = isonn.ResStage(w_in=16, w_out=16, stride=1, transfun=C.ISON.TRANS_FUN, 
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Stage 2: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s2 = isonn.ResStage(w_in=16, w_out=32, stride=2, transfun=C.ISON.TRANS_FUN, 
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Stage 3: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s3 = isonn.ResStage(w_in=32, w_out=64, stride=2, transfun=C.ISON.TRANS_FUN,
                        has_bn=C.ISON.HAS_BN, has_st=C.ISON.HAS_ST, use_srelu=C.ISON.SReLU, d=d)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = isonn.ResHead(w_in=64, nc=C.DATASET.NUM_CLASSES, use_dropout=C.ISON.DROPOUT,
                                dropout_rate=C.ISON.DROPOUT_RATE)

 
    def _network_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if C.ISON.DIRAC_INIT:
                    # the first 7x7 convolution we use pytorch default initialization
                    # and not enforce orthogonality since the large input/output channel difference
                    if m.kernel_size != (7, 7):
                        nn.init.dirac_(m.weight)
                else:
                    # kaiming initialization used for ResNet results
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = (
                    hasattr(m, 'final_bn') and m.final_bn
                )
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def ortho(self):
        ortho_penalty = []
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (7, 7) or m.weight.shape[1] == 3:
                    continue
                o = self.ortho_conv(m)
                cnt += 1
                ortho_penalty.append(o)
        ortho_penalty = sum(ortho_penalty)
        return ortho_penalty

    def ortho_conv(self, m, device='cuda'):
        operator = m.weight
        operand = torch.cat(torch.chunk(m.weight, m.groups, dim=0), dim=1)
        transposed = m.weight.shape[1] < m.weight.shape[0]
        num_channels = m.weight.shape[1] if transposed else m.weight.shape[0]
        if transposed:
            operand = operand.transpose(1, 0)
            operator = operator.transpose(1, 0)
        gram = F.conv2d(operand, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1),
                        stride=m.stride, groups=m.groups)
        identity = torch.zeros(gram.shape).to(device)
        identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(num_channels).repeat(1, m.groups)
        out = torch.sum((gram - identity) ** 2.0) / 2.0
        return out 

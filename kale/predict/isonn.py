# This file is modified by Haiping Lu from https://github.com/HaozhiQi/ISONet/blob/master/isonet/models/isonet.py
# This file is modified from https://github.com/facebookresearch/pycls/blob/master/pycls/models/resnet.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from config import C

def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        'basic_transform': BasicTransform,
        'bottleneck_transform': BottleneckTransform,
    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]


class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, -1.0)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias


# class SharedScale(nn.Module):
#     """Channel-shared scalar"""
#     def __init__(self):
#         super(SharedScale, self).__init__()
#         self.scale = nn.Parameter(torch.ones(1, 1, 1, 1) * C.ISON.RES_MULTIPLIER)

#     def forward(self, x):
#         return x * self.scale


class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, w_in, nc, use_dropout, dropout_rate):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_dropout = use_dropout 
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, w_in, w_out, stride, has_bn, use_srelu, w_b=None, num_gs=1):
        assert w_b is None and num_gs == 1, \
            'Basic transform does not support w_b and num_gs options'
        super(BasicTransform, self).__init__()
        self.has_bn=has_bn
        self.use_srelu=use_srelu
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=stride, padding=1, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.a_bn = nn.BatchNorm2d(w_out)
        self.a_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_out)
        # 3x3, BN
        self.b = nn.Conv2d(
            w_out, w_out, kernel_size=3,
            stride=1, padding=1, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.b_bn = nn.BatchNorm2d(w_out)
            self.b_bn.final_bn = True

        # if C.ISON.HAS_RES_MULTIPLIER:
        #     self.shared_scalar = SharedScale()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, has_bn, use_srelu, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        self.has_bn=has_bn
        self.use_srelu=use_srelu        
        self._construct(w_in, w_out, stride, w_b, num_gs)

    def _construct(self, w_in, w_out, stride, w_b, num_gs):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (str1x1, str3x3) = (1, stride)
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_b, kernel_size=1,
            stride=str1x1, padding=0, bias=not self.has_bn and notself.use_srelu
        )
        if self.has_bn:
            self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_b)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3,
            stride=str3x3, padding=1, groups=num_gs, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_b)
        # 1x1, BN
        self.c = nn.Conv2d(
            w_b, w_out, kernel_size=1,
            stride=1, padding=0, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.c_bn = nn.BatchNorm2d(w_out)
            self.c_bn.final_bn = True

        # if C.ISON.HAS_RES_MULTIPLIER:
        #     self.shared_scalar = SharedScale()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
        self, w_in, w_out, stride, trans_fun, has_bn, has_st, use_srelu, w_b=None, num_gs=1
    ):
        super(ResBlock, self).__init__()
        self.has_bn=has_bn
        self.has_st=has_st
        self.use_srelu=use_srelu               
        self._construct(w_in, w_out, stride, trans_fun, w_b, num_gs)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1,
            stride=stride, padding=0, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.bn = nn.BatchNorm2d(w_out)

    def _construct(self, w_in, w_out, stride, trans_fun, w_b, num_gs):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block and self.has_st:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = trans_fun(w_in, w_out, stride, self.has_bn, self.use_srelu,  w_b, num_gs)
        self.relu = nn.ReLU(True) if not self.use_srelu else SReLU(w_out)

    def forward(self, x):
        if self.proj_block:
            if self.has_bn and self.has_st:
                x = self.bn(self.proj(x)) + self.f(x)
            elif not self.has_bn and self.has_st:
                x = self.proj(x) + self.f(x)
            else:
                x = self.f(x)
        else:
            if self.has_st:
                x = x + self.f(x)
            else:
                x = self.f(x)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, transfun, has_bn, has_st, use_srelu, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        self.transfun=transfun
        self.has_bn = has_bn
        self.has_st = has_st
        self.use_srelu = use_srelu
        self._construct(w_in, w_out, stride, d, w_b, num_gs)

    def _construct(self, w_in, w_out, stride, d, w_b, num_gs):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Retrieve the transformation function
            trans_fun = get_trans_fun(self.transfun)
            # Construct the block
            res_block = ResBlock(
                b_w_in, w_out, b_stride, trans_fun, self.has_bn, self.has_st, self.use_srelu, w_b, num_gs
            )    
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(self, w_in, w_out, has_bn, use_srelu, kernelsize, stride, padding,
                use_maxpool=False, poolksize=3, poolstride=2, poolpadding=1):
        super(ResStem, self).__init__()
        self.has_bn=has_bn
        self.use_srelu=use_srelu        
        self.kernelsize=kernelsize
        self.stride=stride
        self.padding=padding
        self.use_maxpool=use_maxpool        
        self.poolksize=poolksize
        self.poolstride=poolstride
        self.poolpadding=poolpadding
        self._construct(w_in, w_out)
        
        
    def _construct(self, w_in, w_out):
        # 3x3, BN, ReLU for cifar and  7x7, BN, ReLU, maxpool for imagenet
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=self.kernelsize,
            stride=self.stride, padding=self.padding, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(True) if not self.use_srelu else SReLU(w_out)
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=self.poolksize, stride=self.poolstride, padding=self.poolpadding)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x
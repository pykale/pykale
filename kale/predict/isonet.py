"""
The ISONet module, which is based on the ResNet module,
from https://github.com/HaozhiQi/ISONet/blob/master/isonet/models/isonet.py
(based on https://github.com/facebookresearch/pycls/blob/master/pycls/models/resnet.py)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stage depths for ImageNet models
_IN_STAGE_DS = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    46: (3, 4, 12, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    assert name in trans_funs.keys(), "Transformation function '{}' not supported".format(name)
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

    def __init__(self, w_in, net_params):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_dropout = net_params["use_dropout"]
        if self.use_dropout:
            self.dropout = nn.Dropout(p=net_params["dropout_rate"], inplace=True)
        self.fc = nn.Linear(w_in, net_params["nc"], bias=True)

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
        assert w_b is None and num_gs == 1, "Basic transform does not support w_b and num_gs options"
        super(BasicTransform, self).__init__()
        self.has_bn = has_bn
        self.use_srelu = use_srelu
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.a_bn = nn.BatchNorm2d(w_out)
        self.a_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_out)
        # 3x3, BN
        self.b = nn.Conv2d(
            w_out, w_out, kernel_size=3, stride=1, padding=1, bias=not self.has_bn and not self.use_srelu
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
    """Bottleneck transformation: 1x1, 3x3, 1x1, only for very deep networks"""

    def __init__(self, w_in, w_out, stride, has_bn, use_srelu, w_b, num_gs):
        super(BottleneckTransform, self).__init__()
        self.has_bn = has_bn
        self.use_srelu = use_srelu
        self._construct(w_in, w_out, stride, w_b, num_gs)

    def _construct(self, w_in, w_out, stride, w_b, num_gs):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (str1x1, str3x3) = (1, stride)
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_b, kernel_size=1, stride=str1x1, padding=0, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_b)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b,
            w_b,
            kernel_size=3,
            stride=str3x3,
            padding=1,
            groups=num_gs,
            bias=not self.has_bn and not self.use_srelu,
        )
        if self.has_bn:
            self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True) if not self.use_srelu else SReLU(w_b)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=not self.has_bn and not self.use_srelu)
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

    def __init__(self, w_in, w_out, stride, trans_fun, has_bn, has_st, use_srelu, w_b=None, num_gs=1):
        super(ResBlock, self).__init__()
        self.has_bn = has_bn
        self.has_st = has_st
        self.use_srelu = use_srelu
        self._construct(w_in, w_out, stride, trans_fun, w_b, num_gs)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=not self.has_bn and not self.use_srelu
        )
        if self.has_bn:
            self.bn = nn.BatchNorm2d(w_out)

    def _construct(self, w_in, w_out, stride, trans_fun, w_b, num_gs):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block and self.has_st:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = trans_fun(w_in, w_out, stride, self.has_bn, self.use_srelu, w_b, num_gs)
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

    def __init__(self, w_in, w_out, stride, net_params, d, w_b=None, num_gs=1):
        super(ResStage, self).__init__()
        self.transfun = net_params["transfun"]
        self.has_bn = net_params["has_bn"]
        self.has_st = net_params["has_st"]
        self.use_srelu = net_params["use_srelu"]
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
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResStem(nn.Module):
    """Stem of ResNet."""

    def __init__(
        self,
        w_in,
        w_out,
        net_params,
        kernelsize=3,
        stride=1,
        padding=1,
        use_maxpool=False,
        poolksize=3,
        poolstride=2,
        poolpadding=1,
    ):
        super(ResStem, self).__init__()
        self.has_bn = net_params["has_bn"]
        self.use_srelu = net_params["use_srelu"]
        self.kernelsize = kernelsize
        self.stride = stride
        self.padding = padding
        self.use_maxpool = use_maxpool
        self.poolksize = poolksize
        self.poolstride = poolstride
        self.poolpadding = poolpadding
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        # 3x3, BN, ReLU for cifar and  7x7, BN, ReLU, maxpool for imagenet
        self.conv = nn.Conv2d(
            w_in,
            w_out,
            kernel_size=self.kernelsize,
            stride=self.stride,
            padding=self.padding,
            bias=not self.has_bn and not self.use_srelu,
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


class ISONet(nn.Module):
    """ISONet, a modified ResNet model."""

    # def __init__(self, use_dirac=True):
    def __init__(self, net_params):
        super(ISONet, self).__init__()
        # define network structures
        # self._construct()
        self._construct(net_params)
        # initialization
        self._network_init(net_params["use_dirac"])

    # Depth for ResNet, e.g. [3, 4, 6, 3] for ResNet50
    def _construct(self, net_params):
        # Setting for ImageNet image size. To override if different.
        # Retrieve the number of blocks per stage
        (d1, d2, d3, d4) = _IN_STAGE_DS[net_params["depths"]]  # _depths
        # Compute the initial bottleneck width
        # Stem: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.stem = ResStem(w_in=3, w_out=64, net_params=net_params)
        # Stage 1: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s1 = ResStage(w_in=64, w_out=64, stride=1, net_params=net_params, d=d1)
        # Stage 2: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s2 = ResStage(w_in=64, w_out=128, stride=2, net_params=net_params, d=d2)
        # Stage 3: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s3 = ResStage(w_in=128, w_out=256, stride=2, net_params=net_params, d=d3)
        # Stage 4: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s4 = ResStage(w_in=256, w_out=512, stride=2, net_params=net_params, d=d4)
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(w_in=512, net_params=net_params)

    def _network_init(self, use_dirac=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if use_dirac:
                    # the first 7x7 convolution we use pytorch default initialization
                    # and not enforce orthogonality since the large input/output channel difference
                    if m.kernel_size != (7, 7):
                        nn.init.dirac_(m.weight)
                else:
                    # kaiming initialization used for ResNet results
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def ortho(self, device):
        """regularizes the convolution kernel to be (near) orthogonal during training.
        This is called in Trainer.loss of the isonet example.
        """
        ortho_penalty = []
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (7, 7) or m.weight.shape[1] == 3:
                    continue
                o = self.ortho_conv(m, device)
                cnt += 1
                ortho_penalty.append(o)
        ortho_penalty = sum(ortho_penalty)
        return ortho_penalty

    def ortho_conv(self, m, device):
        """regularizes the convolution kernel to be (near) orthogonal during training.

        Args:
            m (nn.module]): [description]
        """
        operator = m.weight
        operand = torch.cat(torch.chunk(m.weight, m.groups, dim=0), dim=1)
        transposed = m.weight.shape[1] < m.weight.shape[0]
        num_channels = m.weight.shape[1] if transposed else m.weight.shape[0]
        if transposed:
            operand = operand.transpose(1, 0)
            operator = operator.transpose(1, 0)
        gram = F.conv2d(
            operand, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1), stride=m.stride, groups=m.groups
        )
        identity = torch.zeros(gram.shape).to(device)
        identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(num_channels).repeat(1, m.groups)
        out = torch.sum((gram - identity) ** 2.0) / 2.0
        return out

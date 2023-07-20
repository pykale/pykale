"""This module implements three different multimodal fusion methods:
1. Concat
2. BimodalInteractionFusion
3. LowRankTensorFusion
Each of these fusion methods are designed to work with input modalities as PyTorch tensors and perform different operations to combine and create a joint representation of the input data.
Reference: https://github.com/pliang279/MultiBench/blob/main/fusions/common_fusions.py
"""

import torch
from torch import nn
from torch.autograd import Variable


class Concat(nn.Module):
    """Concat is a simple PyTorch module for fusing multimodal data by concatenating tensors along dimension 1.
       This fusion method is often used in multimodal learning where data from different modalities (e.g., image, audio) are processed separately and then fused together for further processing or decision making. Each modality data is first flattened from its second dimension onward and then these flattened tensors are concatenated together.
       This approach to fusion maintains the independence of the modalities before the fusion point, allowing the network to learn separate representations for each modality before combining them.
    """

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class BimodalInteractionFusion(nn.Module):
    """ BimodalInteractionFusion is a PyTorch module that performs fusion of two data modalities through a hypernetwork-based interaction mechanism. The 'input_dims' argument specifies the input dimensions of the two modalities. The 'output_dim' argument specifies the output dimension after the fusion. The 'output' argument defines the type of bimodal matrix interactions to be performed, which can be 'matrix', 'vector', or 'scalar'.
        This fusion method  supports three types of bimodal interactions:
            - Matrix: It implements a general hypernetwork mechanism where the interaction is multiplicative. It uses separate weight matrices and biases for the two modalities.
            - Vector: It uses diagonal forms and gating mechanisms, applying element-wise multiplication to combine the modalities.
            - Scalar: It applies scales and biases to the input modalities before combining them.
        This fusion method uses xavier normal distribution for initializing the weight matrices and normal distribution for the biases. It also provides options to clip the parameter values and their gradients within specified ranges to prevent them from exploding or vanishing.
        This fusion approach allows for complex interactions between the modalities and is well-suited for tasks that require the integration of heterogeneous data.
    Args:
        input_dims (int): list or tuple of 2 integers indicating input dimensions of the 2 modalities
        output_dim (int): output dimension after the fusion
        output (str): type of BimodalMatrix Interactions, options from 'matrix','vector','scalar'
        flatten (bool): whether we need to flatten the input modalities
        clip (tuple, optional): clip parameter values, None if no clip
        grad_clip (tuple, optional): clip grad values, None if no clip
        flip (bool): whether to swap the two input modalities in forward function or not
    """

    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        super(BimodalInteractionFusion, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == "matrix":
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        elif output == "vector":
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        elif output == "scalar":
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == "matrix":
            Wprime = torch.einsum("bn, nmd -> bmd", m1, self.W) + self.V  # bmd
            bprime = torch.matmul(m1, self.U) + self.b  # bmd
            output = torch.einsum("bm, bmd -> bd", m2, Wprime) + bprime  # bmd

        elif self.output == "vector":
            Wprime = torch.matmul(m1, self.W) + self.V  # bm
            bprime = torch.matmul(m1, self.U) + self.b  # b
            output = Wprime * m2 + bprime  # bm

        elif self.output == "scalar":
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(
                bprime, self.input_dims[1]
            )
        return output


class LowRankTensorFusion(nn.Module):
    """LowRankTensorFusion is a PyTorch module that performs multimodal fusion using a low-rank tensor-based approach.
       The 'input_dims' argument specifies the input dimensions of each modality. The 'output_dim' argument defines the output dimension after the fusion. The 'rank' argument is a hyperparameter specifying the rank for the low-rank tensor decomposition.
       This fusion method performs fusion by assuming a low-rank structure for the interaction tensor, effectively compressing the interaction space. It leverages a set of low-rank factors, one for each modality, that are learned during training.
       These factors are initialized with xavier normal distribution and are applied to their corresponding modalities during the forward pass. A tensor product is computed across all modalities and their respective factors, resulting in a fused tensor.
       Next, a weighted summation of this fused tensor is computed using fusion weights, followed by the addition of a fusion bias. Both fusion weights and bias are learnable parameters initialized with xavier normal distribution and zero respectively.
       The final output is reshaped to the specified 'output_dim' and returned. If 'flatten' is set to True, each modality is first flattened before concatenation with a ones tensor and the subsequent multiplication with its factor.
       This approach provides an efficient and compact representation for capturing interactions among multiple modalities, making it suitable for tasks involving high-dimensional multimodal data.
    Args:
        input_dims (int): A list or tuple of integers indicating input dimensions of the modalities.
        output_dim (int): output dimension after the fusion.
        rank (int): A hyperparameter specifying the rank for the low-rank tensor decomposition.
        flatten (bool): Boolean to dictate if output should be flattened or not. Default: True

    Note:
        Adapted from https://github.com/Justin1904/Low-rank-Multimodal-Fusion.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        super(LowRankTensorFusion, self).__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(self.rank, input_dim + 1, self.output_dim)).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        batch_size = modalities[0].shape[0]
        fused_tensor = 1
        for modality, factor in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            if self.flatten:
                modality_withones = torch.cat((ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

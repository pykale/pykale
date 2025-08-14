import math

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class RandomLayer(nn.Module):
    """
    The `RandomLayer` is designed to apply random matrix multiplications to a list of input tensors. Each input tensor
    is multiplied by a randomly initialized matrix, and the results are combined through element-wise multiplication.

    Args:
        input_dim_list (list of int): A list of integers representing the dimensionality of each input tensor.
                                      The length of this list determines how many input tensors the layer expects.

        output_dim (int, optional): The dimensionality of the output tensor after the random transformations.
                                    Default is 256.
    """

    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim

        self.random_matrix = nn.ParameterList(
            nn.Parameter(torch.randn(input_dim_list[i], output_dim)) for i in range(self.input_num)
        )

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


class FCNet(nn.Module):
    """
    A simple class for non-linear fully connect network

    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py


    This class creates a fully connected neural network with optional dropout and activation
    functions. Weight normalization is applied to each linear layer.

    Args:
        dims (list of int): A list specifying the input and output dimensions of each layer.
                            For example, [input_dim, hidden_dim1, hidden_dim2, ..., output_dim].
        activation (str, optional): The name of the activation function to use (e.g., 'ReLU', 'Tanh').
                             Default is 'ReLU'. If an empty string is provided, no activation is applied.
        dropout (float, optional): Dropout probability to apply after each layer. Default is 0 (no dropout).

    """

    def __init__(self, dims, activation="ReLU", dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if "" != activation:
                layers.append(getattr(nn, activation)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if "" != activation:
            layers.append(getattr(nn, activation)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

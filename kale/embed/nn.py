import math
import torch
import torch.nn as nn


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
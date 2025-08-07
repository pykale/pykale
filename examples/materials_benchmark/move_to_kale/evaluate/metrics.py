import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import grad
from torch.nn import functional as F


def mean_relative_error(output, target):
    '''
        Calculate the mean relative error between true and predicted values using PyTorch.
        Args:
            output (torch.Tensor): Predicted values.
            target (torch.Tensor): True values. 
        Returns:
            float: Mean relative error.
        '''
        # Epsilon to avoid division by zero if y_true can be zero
    eps = 1e-9
    errors = torch.abs(target - output) / (torch.abs(target) + eps)
    return torch.mean(errors).item()
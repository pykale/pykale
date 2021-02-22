import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    r"""
    The MLP decoder module, which comprises four fully connected neural networks. It's a common decoder for decoding
    drug-target encoding information.

    Args:
        in_dim (int): Dimension of input feature.
        hidden_dim (int): Dimension of hidden layers.
        out_dim (int): Dimension of output layer.
        dropout_rate (float): dropout rate during training.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, 1)
        torch.nn.init.normal_(self.fc4.weight)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

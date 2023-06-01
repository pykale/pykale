"""Two layered mlp classifier.
References: https://github.com/pliang279/MultiBench/blob/main/unimodals/common_models.py
"""

from torch import nn
from torch.nn import functional as F


class MLPClassifier(nn.Module):
    """Initialize multi-layer perceptron for prediction.

    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output layer dimension
        dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

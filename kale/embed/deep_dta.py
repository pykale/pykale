"""
DeepDTA based models for drug-target interaction prediction problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp


class DeepDTAEncoder(nn.Module):
    r"""
    The DeepDTA's CNNs encoder module, which comprises three 1D-convolutional layers and one max-pooling layer. The module
    is applied to encoding drug/target sequence information, and the input should be transformed information with
    integer/label encoding  The original paper is `"DeepDTA: deep drug–target binding affinity prediction"
    <https://academic.oup.com/bioinformatics/article/34/17/i821/5093245>`_ .

    Args:
        num_embeddings (int): Number of embedding labels/categories, depends on the types of encoding sequence.
        embedding_dim (int): Dimensions of embedding labels/categories.
        sequence_length (int): Max length of input sequence.
        num_kernels (int): Number of kernels (filters).
        kernel_length (int): Length of kernel (filter).
    """

    def __init__(self, num_embeddings, embedding_dim, sequence_length, num_kernels, kernel_length):
        super(DeepDTAEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=num_kernels, kernel_size=kernel_length)
        self.conv2 = nn.Conv1d(in_channels=num_kernels, out_channels=num_kernels * 2, kernel_size=kernel_length)
        self.conv3 = nn.Conv1d(in_channels=num_kernels * 2, out_channels=num_kernels * 3, kernel_size=kernel_length)
        self.gmp = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gmp(x)
        x = x.squeeze(2)
        return x


class DrugGCNEncoder(nn.Module):
    r"""
    The GraphDTA's GCNs encoder module, which comprises three graph convolutional layers and one full connected layer.
    The model is a variant of DeepDTA and is applied to encode drug molecule graph information. the original paper
    is  `"GraphDTA: Predicting drug–target binding affinity with graph neural networks"
    <https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btaa921/5942970>`_ .

    Args:
        in_channel (int): Size of each input node.
        out_channel (int): Size of each output node.
        dropout (float): Dropout rate (probability of an element to be zeroed in after full connected layer).
    """
    def __init__(self, in_channel=78, out_channel=128, dropout=0.2):
        super(DrugGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, in_channel)
        self.conv2 = GCNConv(in_channel, in_channel * 2)
        self.conv3 = GCNConv(in_channel * 2, in_channel * 4)
        self.fc = nn.Linear(in_channel * 4, out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class MLPDecoder(nn.Module):
    r"""
    The MLP decoder module, which comprises four full connected neutral network. It's a common decoder for decoding
    drug-target encoding information.

    Args:
        in_dim (int): Dimensions of input feature.
        hidden_dim (int): Dimensions of hidden layers.
        out_dim (int): Dimensions of output layer.
        dropout (float): probability of an element to be zeroed.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, 1)
        torch.nn.init.normal_(self.fc4.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

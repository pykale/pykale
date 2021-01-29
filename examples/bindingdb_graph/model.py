import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.data import dataloader


# GCN based model
class DrugGCNEncoder(nn.Module):
    def __init__(self, in_channel=78, out_channel=128, dropout=0.5):
        super(DrugGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, in_channel)
        self.conv2 = GCNConv(in_channel, in_channel * 2)
        self.conv3 = GCNConv(in_channel * 2, in_channel * 4)
        self.fc = nn.Linear(in_channel * 4, out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)

        x = self.fc(x)
        x = self.dropout(x)

        return x


class TargetConvEncoder(nn.Module):
    def __init__(self, num_target_label=25, embed_dim_label=128, in_channel=1000, out_channel=32, kernel_size=8, output_dim=128):
        super(TargetConvEncoder, self).__init__()
        self.out_channel = out_channel
        self.embedding = nn.Embedding(num_target_label+1, embed_dim_label)
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size)
        self.fc1 = nn.Linear(out_channel*121, output_dim)

    def forward(self, target_label_emb):
        x = self.embedding(target_label_emb)
        x = self.conv1(x)
        # flatten
        x = x.view(-1, self.out_channel * 121)
        x = self.fc1(x)

        return x


class MLPDecoder(nn.Module):
    def __init__(self, in_channel=256, hidden_channel=256, dropout=0.5):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_channel, hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

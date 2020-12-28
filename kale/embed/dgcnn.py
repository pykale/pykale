import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear
from torch_geometric.nn import GCNConv, global_sort_pool


class DGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network (DGCNN) is a GNN architecture for graph classification. The model is from
    the '"an end-to-end deep learning architecture for graph
    classification"<https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>'_ (AAAI 2018) paper. The code is based on
    'SEAL_OGB'<https://github.com/facebookresearch/SEAL_OGB>_ source repo with slight modification.
    Args:
        hidden_channels: size of GCN hidden layers of each feature embedding.
        num_layers: number of graph convolution layers.
        max_z: the maximum number of node labels.
        k: the number of nodes to hold in sort pooling layer for each graph.
        node_embedding: the initial node embedding if set.
    """
    def __init__(self, hidden_channels, num_layers, max_z, k=30, node_embedding=None):
        super(DGCNN, self).__init__()

        self.node_embedding = node_embedding
        self.k = int(k)

        self.max_z = max_z
        self.z_embedding = nn.Embedding(self.max_z, hidden_channels)

        self.convs = nn.ModuleList()
        initial_channels = hidden_channels
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, node_label_index, edge_index, node_batch_index, edge_weight=None, node_id=None):
        """
        Args:
            node_label_index: the label index for each node in graph.
            edge_index: edge index in COO format with shape [2, num_edges].
            node_batch_index: batch index for each node.
            edge_weight: each edge weight if set.
            node_id: node id in graph.
        """
        x = self.z_embedding(node_label_index)
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index, edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # Global sort pooling.
        x = global_sort_pool(x, node_batch_index, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

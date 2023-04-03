from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.embed.gripnet import GripNet
from kale.evaluate.metrics import auprc_auroc_ap
from kale.prepdata.graph_negative_sampling import typed_negative_sampling
from kale.prepdata.supergraph_construct import SuperGraph


class MLPDecoder(nn.Module):
    r"""
    The MLP decoder module, which comprises four fully connected neural networks. It's a common decoder for decoding
    drug-target encoding information.

    Args:
        in_dim (int): the dimension of input feature.
        hidden_dim (int): the dimension of hidden layers.
        out_dim (int): the dimension of output layer.
        dropout_rate (float): the dropout rate during training.
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


class DistMultDecoder(torch.nn.Module):
    """
    Build `DistMult
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf>`_ factorization as GripNet decoder in PoSE dataset.
    Copy-paste with slight modifications from https://github.com/NYXFLOWER/GripNet.

    Args:
        in_channels (int): the dimension of input feature.
        num_edge_type (int): the number of edge types.
    """

    def __init__(self, in_channels: int, num_edge_type: int):
        super(DistMultDecoder, self).__init__()
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(num_edge_type, in_channels))

        self.reset_parameters()

    def forward(self, x, edge_index: torch.Tensor, edge_type: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        """
        Args:
            x: the input node feature embeddings.
            edge_index: the edge index in COO format with shape [2, num_edges].
            edge_type: the one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: whether to use sigmoid function or not.
        """
        value = (x[edge_index[0]] * x[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_channels))

    def __repr__(self) -> str:
        return "{}: DistMultLayer(in_channels={}, num_relations={})".format(
            self.__class__.__name__, self.in_channels, self.num_edge_type
        )


class GripNetLinkPrediction(pl.LightningModule):
    """
    Build GripNet-DistMult (encoder-decoder) model for link prediction.

    Args:
        supergraph (SuperGraph): the input supergraph.
        learning_rate (float): the learning rate for training.
        epsilon (float, optional): a small number in loss function to improve numerical stability. Defaults to 1e-13.
    """

    def __init__(self, supergraph: SuperGraph, learning_rate: float, epsilon: float = 1e-13):
        super().__init__()

        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.encoder = GripNet(supergraph)
        self.decoder = self.__init_decoder__()

    def __init_decoder__(self) -> DistMultDecoder:
        in_channels = self.encoder.out_channels
        supergraph = self.encoder.supergraph
        task_supervertex_name = supergraph.topological_order[-1]
        num_edge_type = supergraph.supervertex_dict[task_supervertex_name].num_edge_type

        # get the number of nodes on the task-associated supervertex
        self.num_task_nodes = supergraph.supervertex_dict[task_supervertex_name].num_node

        return DistMultDecoder(in_channels, num_edge_type)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_type_range: torch.Tensor) -> Tuple:
        x = self.encoder()

        pos_score = self.decoder(x, edge_index, edge_type)
        pos_loss = -torch.log(pos_score + self.epsilon).mean()

        edge_index = typed_negative_sampling(edge_index, self.num_task_nodes, edge_type_range)

        neg_score = self.decoder(x, edge_index, edge_type)
        neg_loss = -torch.log(1 - neg_score + self.epsilon).mean()

        loss = pos_loss + neg_loss

        # compute averaged metric scores over edge types
        num_edge_type = edge_type_range.shape[0]
        record = []
        for i in range(num_edge_type):
            start, end = edge_type_range[i]
            pos_score_this_type, neg_score_this_type = pos_score[start:end], neg_score[start:end]

            score = torch.cat([pos_score_this_type, neg_score_this_type])
            target = torch.cat([torch.ones(pos_score_this_type.shape[0]), torch.zeros(neg_score_this_type.shape[0])])

            record.append(list(auprc_auroc_ap(target, score)))
        auprc, auroc, ave_precision = np.array(record).mean(axis=0)

        return loss, auprc, auroc, ave_precision

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def __step__(self, batch, mode="train"):
        edge_index, edge_type, edge_type_range = batch
        loss, auprc, auroc, ap = self.forward(
            edge_index.reshape((2, -1)), edge_type.flatten(), edge_type_range.reshape((-1, 2))
        )

        if mode == "train" or "val":
            self.log(f"{mode}_loss", loss)
        else:
            self.log(f"{mode}_auprc", auprc)
            self.log(f"{mode}_auroc", auroc)
            self.log(f"{mode}_ap@50", ap)

        return loss

    def training_step(self, batch, batch_idx):
        return self.__step__(batch)

    def validation_step(self, batch, batch_idx):
        return self.__step__(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.__step__(batch, mode="test")

    def __repr__(self) -> str:
        return "{}: \nEncoder: {} ModuleDict(\n{})\n Decoder: {}".format(
            self.__class__.__name__, self.encoder.__class__.__name__, self.encoder.supervertex_module_dict, self.decoder
        )

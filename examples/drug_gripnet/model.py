import numpy as np
import torch
from torch import nn
from utils import negative_sampling

from kale.embed.gripnet import TypicalGripNetEncoder


class MultiRelaInnerProductDecoder(nn.Module):
    """
    Build `DistMult
    <https://arxiv.org/abs/1412.6575>`_ factorization as GripNet decoder in PoSE dataset.
    """

    def __init__(self, in_dim, num_et):
        super(MultiRelaInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        """
        Args:
            z: input node feature embeddings.
            edge_index: edge index in COO format with shape [2, num_edges].
            edge_type: The one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: use sigmoid function or not.
        """
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


class GripNet(nn.Module):
    """
    Build GripNet-DistMult (Encoder-Decoder) model for PoSE link prediction.
    """

    def __init__(
        self,
        gene_channels_list,
        gd_channels_list,
        drug_channels_list,
        num_drug_nodes,
        num_gene_nodes,
        num_drug_edge_relations,
    ):
        """
        Parameter meanings explained in kale.embed.gripnet module.
        """
        super(GripNet, self).__init__()
        self.num_drug_nodes = num_drug_nodes
        self.num_gene_nodes = num_gene_nodes
        self.gn = TypicalGripNetEncoder(
            gene_channels_list,
            gd_channels_list,
            drug_channels_list,
            num_drug_nodes,
            num_gene_nodes,
            num_drug_edge_relations,
        )
        self.dmt = MultiRelaInnerProductDecoder(sum(drug_channels_list), num_drug_edge_relations)

    def forward(
        self,
        gene_x,
        gene_edge_index,
        gene_edge_weight,
        gd_edge_index,
        drug_index,
        drug_edge_types,
        drug_edge_range,
        device,
    ):
        z = self.gn(
            gene_x, gene_edge_index, gene_edge_weight, gd_edge_index, drug_index, drug_edge_types, drug_edge_range
        )
        pos_index = drug_index
        neg_index = negative_sampling(drug_index, self.num_drug_nodes).to(device)
        pos_score = self.dmt(z, pos_index, drug_edge_types)
        neg_score = self.dmt(z, neg_index, drug_edge_types)
        return pos_score, neg_score

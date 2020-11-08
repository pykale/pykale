import torch
import numpy as np
from torch.nn import Parameter, Module
from kale.embed.gripnet import TypicalGripNetEncoder
from utils import negative_sampling


class MultiRelaInnerProductDecoder(Module):
    """
    Build DisMult factorization as GripNet decoder in PoSE dataset.
    """
    def __init__(self, in_dim, num_et):
        super(MultiRelaInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(num_et, in_dim))

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


class GripNet(Module):
    """
    Build GripNet-DisMult (Encoder-Decoder) model for PoSE link prediction.
    """
    def __init__(self, gg_nhids_gcn, gd_out, dd_nhids_gcn, n_d_node, n_g_node, n_dd_edge_type):
        """
        Parameter meanings explained in kale.embed.gripnet module
        """
        super(GripNet, self).__init__()
        self.n_d_node = n_d_node
        self.n_g_node = n_g_node
        self.gn = TypicalGripNetEncoder(gg_nhids_gcn, gd_out, dd_nhids_gcn, n_d_node, n_g_node, n_dd_edge_type)
        self.dmt = MultiRelaInnerProductDecoder(sum(dd_nhids_gcn), n_dd_edge_type)

    def forward(self, g_feat, gg_edge_index, gg_edge_weight, gd_edge_index, dd_idx, dd_et, dd_range, device):
        z = self.gn(g_feat, gg_edge_index, gg_edge_weight, gd_edge_index, dd_idx, dd_et, dd_range)
        pos_index = dd_idx
        neg_index = negative_sampling(dd_idx, self.n_d_node).to(device)
        pos_score = self.dmt(z, pos_index, dd_et)
        neg_score = self.dmt(z, neg_index, dd_et)
        return pos_score, neg_score

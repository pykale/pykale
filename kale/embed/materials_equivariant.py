# This file contains the implementation of modules that embeds crystal structures or alike graph data
# in a materials science context, specifically for equivariant neural networks.

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


class rbf_emb(nn.Module):
    """
    Wrapper for RBF embedding using ExpNormalSmearing + CosineCutoff backend.
    Keeps the same signature as LEFTNet's original rbf_emb.
    The following code is from the "LEFTNet repository<https://github.com/yuanqidu/LeftNet>".
    """

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbf_layer = ExpNormalSmearing(
            cutoff_lower=0.0,
            cutoff_upper=rbound_upper,
            num_rbf=num_rbf,
            trainable=rbf_trainable,
        )

    def reset_parameters(self):
        self.rbf_layer.reset_parameters()

    def forward(self, dist):
        return self.rbf_layer(dist)


class NeighborEmb(MessagePassing):

    """
    Embeds the atom features and neighbor features into a higher-dimensional space.
    The following code is from the "LEFTNet repository<https://github.com/yuanqidu/LeftNet>".

    Args:
        hid_dim (int): Dimension of the hidden space.
        input_dim (int): Dimension of the input features.
    """

    def __init__(self, hid_dim: int, input_dim: int):
        super(NeighborEmb, self).__init__(aggr="add")
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(self, z, edge_index, embs):
        """Atom Embedding + Neighborhours Embedding"""
        z_emb = self.ln_emb(self.fc(z))  # shape: (num_nodes, hidden_channels)
        s_neighbors = self.propagate(edge_index, x=z_emb, norm=embs)
        z_emb = z_emb + s_neighbors
        # s = s + s_neighbors
        return z_emb  # shape: (num_nodes, hidden_channels)

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    """
    Message passing layer that computes the new features for each atom based on its neighbors.
    """

    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr="add")
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.LayerNorm(hid_dim, elementwise_affine=False), nn.SiLU()
        )

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    """
    Equivariant message passing layer that aggregates features from neighboring atoms.
    This layer uses a combination of atom features, neighbor features, and radial basis functions
    to compute the new features for each atom.
    The following code is from the "LEFTNet repository<https://github.com/yuanqidu/LeftNet>".

    Args:
        hidden_channels (int): Dimension of the hidden space.
        num_radial (int): Number of radial basis functions to use.
    """

    def __init__(
        self,
        hidden_channels,
        num_radial,
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels
        self.num_radial = num_radial
        self.dir_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3),
        )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, weight, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    """
    Frame transition encoding module from `LEFTNet<https://openreview.net/pdf?id=hWPNYWkYPN>`.
    An equivariant message-passing module that encodes local geometric information by scalarizing tensor-valued edge features in a node-wise frame,
    transforming them via an MLP, and re-tensorizing to update both invariant and tensor features without information loss.

    Args:
        hidden_channels (int): Dimension of the hidden space.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, node_frame):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        # # # scalrization = torch.sum(vec1.unsqueeze(2) * node_frame.unsqueeze(-1), dim=1)
        # # # scalrization[:, 1, :] = torch.abs(scalrization[:, 1, :].clone())
        scalar = torch.sqrt(torch.sum(vec1**2, dim=-2) + 1e-10)

        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class EquiOutput(nn.Module):
    """
    Output layer for equivariant neural networks.
    This layer processes the output from the last message-passing layer and produces a vector representation.

    Args:
        hidden_channels (int): Dimension of the hidden space.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(hidden_channels, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in `Quantum-chemical insights from deep tensor neural networks<https://www.nature.com/articles/ncomms13890>`:
    Equivariant message passing for the prediction of tensorial properties and molecular spectra.

    Args:
        hidden_channels (int): Dimension of the hidden space.
        out_channels (int): Dimension of the output space.
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


# Implementation from TensorNet
# https://github.com/torchmd/torchmd-net
class ExpNormalSmearing(nn.Module):
    """
    Exponentially normal smearing function for radial basis function (RBF) embeddings.
    This function is used to create a smooth representation of distances between atoms in a crystal structure.
    The following code is from the `TensorNet<https://github.com/torchmd/torchmd-net>`_ repository.

    Args:
        cutoff_lower (float): Lower bound of the cutoff distance.
        cutoff_upper (float): Upper bound of the cutoff distance.
        num_rbf (int): Number of radial basis functions to use.
        trainable (bool): Whether the parameters of the RBF are trainable.
        dtype (torch.dtype): Data type for the parameters.
    """

    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower, dtype=self.dtype))
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class CosineCutoff(nn.Module):
    """Cosine cutoff function for radial basis function (RBF) embeddings.
    This function is used to apply a smooth cutoff to the RBF embeddings based on the distance between atoms.
    The following code is from the `TensorNet<https://github.com/torchmd/torchmd-net>`_ repository.
    Args:
        cutoff_lower (float): Lower bound of the cutoff distance.
        cutoff_upper (float): Upper bound of the cutoff distance.
    """

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: Tensor) -> Tensor:
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi * (2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs


from math import pi


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import radius_graph

from torch_scatter import scatter

from kale.embed.materials_equivariant import (
    EquiMessagePassing,
    FTE,
    NeighborEmb,
    rbf_emb,
    S_vector,
)

class LEFTNetZ(nn.Module):
    def __init__(
        self,
        atom_fea_dim,
        num_targets,
        otf_graph=False,
        use_pbc=True,
        output_dim=0,
        cutoff=6.0,
        num_layers=4,
        readout="mean",
        hidden_channels=128,
        num_radial=96,
        y_mean=0,
        y_std=1,
        eps=1e-10,
    ):
        """
        LEFTNet-Z model for predicting properties of materials based on their crystal structure.
        Equivariant GNN for crystal graphs with periodic boundary conditions (PBC) support. It builds a radius graph from atomic positions, uses RBF distance
        encodings and equivariant message passing. Outputs graph-level properties via readout.

        The code is implemented based on the paper `A new perspective on building efficient and expressive
        3D equivariant graph neural networks<https://openreview.net/pdf?id=hWPNYWkYPN>`

        Args:
            atom_fea_dim (int): Input atomic feature dimension (e.g., size of Z/property encoding).
            num_targets (int): Graph-level output dimension (if ``output_dim==0``, this value is used).
            otf_graph (bool): Whether to use on-the-fly graph construction. (Stored but unused here.)
            use_pbc (bool): Whether to enable periodic boundary conditions. (Stored but unused here.)
            output_dim (int): If > 0, overrides ``num_targets`` for the final layer size. Default: 0.
            cutoff (float): Radius cutoff (Å) for neighbor search. Default: 6.0.
            num_layers (int): Number of equivariant message passing blocks. Default: 4.
            readout (str): Readout reducer: {"mean","sum",...}. Default: "mean".
            hidden_channels (int): Hidden channel size for invariant/equivariant states. Default: 128.
            num_radial (int): Number of RBFs for encoding distances. Default: 96.
            y_mean (float): Target normalization mean. Default: 0.
            y_std (float): Target normalization std. Default: 1.
            eps (float): Numerical epsilon to avoid division-by-zero. Default: 1e-10.
        """
        super(LEFTNetZ, self).__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.eps = eps
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.num_targets = num_targets
        self.readout = readout

        self.use_pbc = use_pbc
        self.otf_graph = otf_graph

        self.input_dim = atom_fea_dim

        self.radial_emb = rbf_emb(num_radial, cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels), nn.SiLU(inplace=True), nn.Linear(hidden_channels, hidden_channels)
        )

        self.neighbor_emb = NeighborEmb(self.hidden_channels, self.input_dim)

        self.S_vector = S_vector(hidden_channels)

        self.lin = nn.Sequential(
            nn.Linear(3, hidden_channels // 4), nn.SiLU(inplace=True), nn.Linear(hidden_channels // 4, 1)
        )

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(EquiMessagePassing(hidden_channels, num_radial).jittable())
            self.FTEs.append(FTE(hidden_channels))

        if output_dim != 0:
            self.num_targets = output_dim

        self.last_layer = nn.Linear(hidden_channels, self.num_targets)
        # self.out_forces = EquiOutput(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def prop_setup(self, data, device):
        """
        Setup for the property encoding. for LEFTNet-Z, only the atomic number is used as the input feature.
        num_classes = 95 for the first 95 elements are used in the periodic table.
        """
        z = data.atom_num.long().to(device)
        z_encoded = F.one_hot(z, num_classes=95).float()
        return z_encoded

    # @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        device = next(self.parameters()).device
        pos = data.positions.to(device)
        batch = data.batch_idx.to(device)
        z = self.prop_setup(data, device)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
        j, i = edge_index
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        radial_emb = self.radial_emb(dist)  # RBF, shape: (num_edges, num_radial=32), embedding for raw distance
        radial_hidden = self.radial_lin(radial_emb)  # MLP
        rbounds = 0.5 * (
            torch.cos(dist * pi / self.cutoff) + 1.0
        )  # for soft cutoff, smooth decay, value closer to 1 means stronger relationship, closer to 0 means weaker

        radial_emb = radial_emb.to(device)
        radial_hidden = radial_hidden.to(device)
        rbounds = rbounds.to(device)

        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden  # further stengthen the representation of radial_emb

        # init invariant node features
        # shape: (num_nodes, hidden_channels)
        s = self.neighbor_emb(
            z, edge_index, radial_hidden
        )  # z (num_nodes, atom_encoding), z_emb (num_nodes, hidden_channels); s=z_emb+neighbor_emb, shape: (num_nodes, hidden_channels)

        # init equivariant node features
        # shape: (num_nodes, 3, hidden_channels)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        # build edge-wise frame
        edge_diff = vecs
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)  # normalize the edge_diff
        edge_cross = torch.cross(pos[i], pos[j])
        edge_cross = edge_cross / ((torch.sqrt(torch.sum((edge_cross) ** 2, 1).unsqueeze(1))) + self.eps)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # shape: (num_edges, 3, 3)
        edge_frame = torch.cat(
            (edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1
        )  # normalised edge, edge_cross, edge_vertical

        node_frame = 0

        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        scalar4 = (
            self.lin(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)

        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)

        for i in range(self.num_layers):
            ds, dvec = self.message_layers[i](s, vec, edge_index, radial_emb, edge_weight, edge_diff)

            s = s + ds
            vec = vec + dvec

            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec, node_frame)

            s = s + ds
            vec = vec + dvec

        s = self.last_layer(s)
        s = scatter(s, batch, dim=0, reduce=self.readout)
        s = s * self.y_std + self.y_mean
        return s

    def forward(self, data):
        return self._forward(data)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class LEFTNetProp(LEFTNetZ):
    """
    LEFTNet with property-based atomic encoding. Only differs in ``prop_setup``:
    it uses ``data.atom_fea`` (precomputed atomic properties) instead of Z one-hot.
    """

    def __init__(
        self,
        atom_fea_dim,
        num_targets,
        otf_graph=False,
        use_pbc=True,
        output_dim=0,
        cutoff=8.0,
        num_layers=4,
        readout="mean",
        hidden_channels=128,
        num_radial=96,
        y_mean=0,
        y_std=1,
        eps=1e-10,
    ):
        super(LEFTNetProp, self).__init__(
            atom_fea_dim=atom_fea_dim,
            num_targets=num_targets,
            otf_graph=otf_graph,
            use_pbc=use_pbc,
            output_dim=output_dim,
            cutoff=cutoff,
            num_layers=num_layers,
            readout=readout,
            hidden_channels=hidden_channels,
            num_radial=num_radial,
            y_mean=y_mean,
            y_std=y_std,
            eps=eps,
        )

    def prop_setup(self, data, device):
        """
        Setup for the property encoding. for LEFTNet-Prop, a one-hot encoding contains atomic properties is used as the atomic feature..
        """

        return data.atom_fea.to(device)


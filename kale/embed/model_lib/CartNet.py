
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_cluster import radius_graph

from torch_scatter import scatter

from kale.embed.materials_equivariant import (
    CosineCutoff,

    ExpNormalSmearing,

)



class CartNet(torch.nn.Module):
    """

    CartNet model from Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation.
    This is an implementation of the paper "Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation<https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00352g>"
    The code is adapted from the `CartNet repository<https://github.com/imatge-upc/CartNet/tree/main>`.
    Args:
        dim_in (int): Dimensionality of the input features.
        dim_rbf (int): Dimensionality of the radial basis function embeddings.
        num_layers (int): Number of CartNet layers in the model.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 5.0.
        invariant (bool, optional): If `True`, enforces rotational invariance in the encoder. Default is `False`.
        temperature (bool, optional): If `True`, includes temperature information in the encoder. Default is `True`.
        use_envelope (bool, optional): If `True`, applies an envelope function to the interactions. Default is `True`.
        cholesky (bool, optional): If `True`, uses a Cholesky head for the output. If `False`, uses a scalar head. Default is `True`.
    Methods:
        forward(batch):
            Performs a forward pass of the model.
            Args:
                batch: A batch of input data.
            Returns:
                pred: The model's predictions.
                true: The ground truth values corresponding to the input batch.
    """

    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True,
        use_envelope: bool = True,
        atom_types: bool = True,
    ):
        super().__init__()

        self.encoder = GeometricGraphEncoder(
            dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, temperature=temperature, atom_types=atom_types
        )
        self.dim_in = dim_in

        layers = []
        for _ in range(num_layers):
            layers.append(
                CartNetLayer(
                    dim_in=dim_in,
                    use_envelope=use_envelope,
                    radius=radius,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        self.head = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.SiLU(inplace=True),
            pyg_nn.Linear(dim_in // 2, 1, bias=True),
        )

    def forward(self, batch):
        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)
        dim_size = int(batch.batch_idx.max().item() + 1)
        x = self.head(batch.x)
        pred = scatter(x, batch.batch_idx.to(x.device), dim=0, reduce="mean", dim_size=dim_size)

        return pred, batch.target


class GeometricGraphEncoder(nn.Module):
    """
    Geometry-aware graph encoder.

    Builds edges from 3D coordinates using a radius cutoff, encodes node
    features from categorical types or provided embeddings, and encodes
    edge features from radial basis functions (RBF) and optional direction
    vectors. Supports an invariant mode (distance-only edges) and a
    directional mode for equivariant message passing.

    Args:
        dim_in (int): Output dimension of encoded node features.
        dim_rbf (int): Number of radial basis functions for distance encoding.
        radius (float): Cutoff distance for neighbor search.
        invariant (bool): If True, edges use distances only. If False,
            append unit direction vectors.
        temperature (bool): If True, include temperature in node encoding.
        use_node_type (bool): If True, embed node types (e.g., atomic numbers).

    Inputs (batch):
        - `pos` (Tensor [N, 3]): Cartesian coordinates for nodes.
        - Optional `node_type` (LongTensor [N]): Categorical node IDs (e.g., atomic numbers).
        - Optional `x` (Tensor [N, F]): Pre-computed node features.
        - Optional `temperature` (Tensor): Per-node or per-graph temperatures.
        - `batch` (LongTensor [N]): Graph IDs for each node.

    Outputs:
        - `x` (Tensor [N, dim_in]): Encoded node features.
        - `edge_index` (LongTensor [2, E]): Graph connectivity.
        - `edge_attr` (Tensor [E, Fe]): Encoded edge features.
        - `cart_dist` (Tensor [E]): Pairwise distances.
        - `cart_dir` (Tensor [E, 3]): Direction vectors (if invariant=False).
    """

    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True,
        atom_types: bool = True,
    ):
        super(GeometricGraphEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_rbf = dim_rbf
        self.radius = radius
        self.invariant = invariant
        self.temperature = temperature
        self.atom_types = atom_types
        if self.atom_types:
            self.embedding = nn.Embedding(119, self.dim_in * 2)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        elif not self.temperature:
            self.embedding = nn.Embedding(1, self.dim_in)

        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in * 2, bias=True)
        elif self.atom_types:
            self.bias = nn.Parameter(torch.zeros(self.dim_in * 2))
        self.activation = nn.SiLU(inplace=True)

        if self.temperature or self.atom_types:
            self.encoder_atom = nn.Sequential(
                self.activation, pyg_nn.Linear(self.dim_in * 2, self.dim_in), self.activation
            )
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3

        self.encoder_edge = nn.Sequential(
            pyg_nn.Linear(dim_edge, self.dim_in * 2),
            self.activation,
            pyg_nn.Linear(self.dim_in * 2, self.dim_in),
            self.activation,
        )

        self.rbf = ExpNormalSmearing(0.0, radius, dim_rbf, False)

    def forward(self, batch):
        batch.device = next(self.parameters()).device
        data = batch.atom_num.long().to(batch.device)
        batch_idx = batch.batch_idx.to(batch.device)
        pos = batch.positions.to(batch.device)

        batch.edge_index = radius_graph(pos, r=self.radius, batch=batch_idx, max_num_neighbors=1000)
        j, i = batch.edge_index
        vec = pos[j] - pos[i]
        dist = vec.norm(dim=-1)
        batch.cart_dist = dist
        batch.cart_dir = vec / dist.unsqueeze(-1)

        x = self.embedding(data) + self.bias

        batch.x = self.encoder_atom(x)

        if self.invariant:  # cfg.invariant is False
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch


class CartNetLayer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the CartNet architecture.
    Args:
        dim_in (int): Dimension of the input node features.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 8.0.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.

    """

    def __init__(
        self,
        dim_in: int,
        radius: float = 8.0,
        use_envelope: bool = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True)
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in * 3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in * 3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )

        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, radius)

    def forward(self, batch):
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist

        x_in = x
        e_in = e

        x, e = self.propagate(
            edge_index,
            Xx=x,
            Ee=e,
            He=dist,
        )

        batch.x = self.activation(x) + x_in

        batch.edge_attr = e_in + e

        return batch

    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))

        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1) * e_ij
        else:
            sigma_ij = e_ij

        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))

        out = scatter(sigma_ij * sender, index, 0, None, dim_size, reduce="sum")

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)

        e_out = self.e
        del self.e

        return x, e_out

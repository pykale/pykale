
import torch
import torch.nn as nn


class CGCNNEncoderLayer(nn.Module):
    r"""
    CGCNN-style edge-gated graph convolution for crystal graphs.
    An implementation of paper `Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of
    Material Properties<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.

    Args:
        atom_fea_len (int): Node/atom feature dimension.
        nbr_fea_len (int):  Edge/neighbor feature dimension.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(CGCNNEncoderLayer, self).__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        num_atoms, max_nbr = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(num_atoms, max_nbr, -1), atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, total_gated_fea.shape[-1])).view(num_atoms, max_nbr, -1)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGCN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    An implementation of paper `Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of
    Material Properties<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    Each crystal is provided as:
      - atom features ``data.atom_fea``,
      - dense neighbor lists ``data.nbr_fea_idx``,
      - neighbor edge features ``data.nbr_fea`` (e.g., distance embeddings).

    Args:
        orig_atom_fea_len (int): Input atom feature dim.
        nbr_fea_len (int): Neighbor/edge feature dim.
        atom_fea_len (int): Hidden atom feature dim after embedding. Default: 64.
        n_conv (int): Number of CGCNN layers. Default: 3.
        h_fea_len (int): Hidden dim before output. Default: 128.
        n_h (int): # of hidden FC layers after pooling. Default: 1.
        classification (bool): If True, 2-way classification; else regression.
    """

    def __init__(
        self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, classification=False
    ):
        super(CrystalGCN, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [CGCNNEncoderLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax()
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        device = next(self.parameters()).device
        # atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
        atom_fea = data.atom_fea.to(device)
        nbr_fea = data.nbr_fea.to(device)
        nbr_fea_idx = data.nbr_fea_idx.to(device)
        crystal_atom_idx = data.crystal_atom_idx

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)  # add position information
        # or here posi_fea = fc()
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

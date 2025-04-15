# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
Python implementation of DrugBAN model for predicting drug-protein interactions using PyTorch.

This includes a Graph Convolutional Network (GCN) for extracting features from molecular graphs (drugs),
and a Convolutional Neural Network (CNN) for extracting features from protein sequences.

These features are fused using a Bilinear Attention Network (BAN) layer and passed through an MLP classifier
to predict interactions.
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GCNConv


class DrugBAN(nn.Module):
    """
    A neural network model for predicting drug-protein interactions.

    The DrugBAN model integrates a Graph Convolutional Network (GCN) for extracting features
    from molecular graphs (drugs) and a Convolutional Neural Network (CNN) for extracting
    features from protein sequences. These features are then fused using a Bilinear Attention
    Network (BAN) layer and passed through an MLP classifier to predict interactions.

    Args:
        config (dict): A dictionary containing the configuration parameters for the model.

    Reference:
    Bai, P., Miljković, F., John, B. et al. Interpretable bilinear attention network with domain
    adaptation improves drug–target prediction. Nat Mach Intell 5, 126–136 (2023).
    """

    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        self.drug_extractor = MolecularGCN(
            in_feats=drug_in_feats, dim_embedding=drug_embedding, padding=drug_padding, hidden_feats=drug_hidden_feats
        )
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name="h_mat",
            dim=None,
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    """
    A molecular feature extractor using a Graph Convolutional Network (GCN).

    This class implements a GCN to extract features from molecular graphs. It includes an initial
    linear transformation followed by a series of graph convolutional layers. The output is a
    fixed-size feature vector for each molecule.

    Args:
        in_feats (int): Number of input features each node has.
        dim_embedding (int): Dimensionality of the embedding space after the initial linear transformation.
        padding (bool): Whether to apply padding (set certain weights to zero).
        hidden_feats (list of int): A list specifying the number of hidden units for each GCN layer.
        activation (callable, optional): Activation function to apply after each GCN layer.
    """

    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            # If padding is enabled, set the last row of the weight matrix to zeros (for  any padded (dummy) nodes)
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        self.gcn_layers = nn.ModuleList()
        self.activations = []
        prev_dim = dim_embedding
        for hidden_dim in hidden_feats:
            self.gcn_layers.append(GCNConv(prev_dim, hidden_dim))
            self.activations.append(activation)
            prev_dim = hidden_dim

        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        x, edge_index = batch_graph.x, batch_graph.edge_index
        x = self.init_transform(x)

        for gcn_layer, activation in zip(self.gcn_layers, self.activations):
            x = gcn_layer(x, edge_index)
            if activation is not None:
                x = activation(x)

        # Expect all graphs to be padded to the same number of nodes
        batch_size = batch_graph.num_graphs
        x = x.view(batch_size, -1, self.output_feats)

        return x


class ProteinCNN(nn.Module):
    """
    A protein feature extractor using Convolutional Neural Networks (CNNs).

    This class extracts features from protein sequences using a series of 1D convolutional layers.
    The input protein sequence is first embedded and then passed through multiple convolutional
    and batch normalization layers to produce a fixed-size feature vector.

    Args:
        embedding_dim (int): Dimensionality of the embedding space for protein sequences.
        num_filters (list of int): A list specifying the number of filters for each convolutional layer.
        kernel_size (list of int): A list specifying the kernel size for each convolutional layer.
        padding (bool): Whether to apply padding to the embedding layer.
    """

    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        # self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    """
    A multilayer perceptron (MLP) decoder for processing feature vectors into output predictions.

    The `MLPDecoder` class implements a four-layer fully connected neural network with batch normalization
    and ReLU activations.

    Args:
        in_dim (int): Dimensionality of the input feature vector.
        hidden_dim (int): Dimensionality of the hidden layers.
        out_dim (int): Dimensionality of the output from the third layer, which can be seen as the
                       final hidden representation before the classification layer.
        binary (int, optional): Number of output classes in the final classification layer.
                                Default is 1, which is typically used for binary classification.

    """

    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class RandomLayer(nn.Module):
    """
    The `RandomLayer` is designed to apply random matrix multiplications to a list of input tensors. Each input tensor
    is multiplied by a randomly initialized matrix, and the results are combined through element-wise multiplication.

    Args:
        input_dim_list (list of int): A list of integers representing the dimensionality of each input tensor.
                                      The length of this list determines how many input tensors the layer expects.

        output_dim (int, optional): The dimensionality of the output tensor after the random transformations.
                                    Default is 256.
    """

    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class BANLayer(nn.Module):
    """
    The bilinear Attention Network (BAN) layer is designed to apply bilinear attention between two feature sets (`v` and `q`),
    which could represent features extracted from drugs and proteins, respectively. This layer
    enables the interaction between these two sets of features, allowing the model to learn
    joint representations that can be used for downstream tasks like predicting drug-protein
    interactions.

    Args:
        v_dim (int): Dimensionality of the first input feature set (`v`).
        q_dim (int): Dimensionality of the second input feature set (`q`).
        h_dim (int): Dimensionality of the hidden layer used in the bilinear attention mechanism.
        h_out (int): Number of output heads in the bilinear attention mechanism.
        act (str, optional): Activation function to use in the fully connected networks for `v` and `q`.
                             Default is "ReLU".
        dropout (float, optional): Dropout rate to apply after each layer in the fully connected networks.
                                   Default is 0.2.
        k (int, optional): Number of attention maps to generate (used in pooling). Default is 3.
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act="ReLU", dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """
    A simple class for non-linear fully connect network

    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py


    This class creates a fully connected neural network with optional dropout and activation
    functions. Weight normalization is applied to each linear layer.

    Args:
        dims (list of int): A list specifying the input and output dimensions of each layer.
                            For example, [input_dim, hidden_dim1, hidden_dim2, ..., output_dim].
        act (str, optional): The name of the activation function to use (e.g., 'ReLU', 'Tanh').
                             Default is 'ReLU'. If an empty string is provided, no activation is applied.
        dropout (float, optional): Dropout probability to apply after each layer. Default is 0 (no dropout).

    """

    def __init__(self, dims, act="ReLU", dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if "" != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if "" != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

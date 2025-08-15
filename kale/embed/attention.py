# Created by Raivo Koot from modifying https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from kale.embed.nn import FCNet


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the NIPS2017 paper 'Attention Is All You Need' about Transformers
    (https://arxiv.org/abs/1706.03762).
    Essentially, for all timesteps in a given sequence, adds information about the relative temporal location of a
    timestep directly into the features of that timestep, and then returns this slightly-modified, same-shape sequence.

    args:
        d_model: The number of features that each timestep has (required).
        max_len: The maximum sequence length that the positional encodings should support (required).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        self.scaling_term = math.sqrt(d_model)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        """
        Expects input of shape (sequence_length, batch_size, num_features) and returns output of the same shape.
        sequence_length is at most allowed to be self.max_len and num_features is expected to be exactly self.d_model.

        Args:
            x: a sequence input of shape (sequence_length, batch_size, num_features) (required).
        """
        x = x * self.scaling_term  # make embedding relatively larger than positional encoding
        x = x + self.pe[: x.size(0), :]
        return x


class BANLayer(nn.Module):
    """
    The bilinear Attention Network (BAN) layer is designed to apply bilinear attention between two feature sets
    (`v` and `q`), which could represent features extracted from drugs and proteins, respectively. This layer
    enables the interaction between these two sets of features, allowing the model to learn joint representations
    that can be used for downstream tasks like predicting drug-protein interactions.

    Args:
        input_v_dim (int): Dimensionality of the first input "value" feature set (`v`).
        input_q_dim (int): Dimensionality of the second input "query" feature set (`q`).
        hidden_dim (int): Dimensionality of the hidden layer used in the bilinear attention mechanism.
        num_out_heads (int): Number of output heads in the bilinear attention mechanism.
        activation (str, optional): Activation function to use in the fully connected networks for value (`v`) and
                                    query (`q`). Default is "ReLU".
        dropout (float, optional): Dropout rate to apply after each layer in the fully connected networks.
                                   Default is 0.2.
        num_att_maps (int, optional): Number of attention maps to generate (used in pooling). Default is 3.
    """

    def __init__(
        self, input_v_dim, input_q_dim, hidden_dim, num_out_heads, activation="ReLU", dropout=0.2, num_att_maps=3
    ):
        super().__init__()

        self.c = 32
        self.num_att_maps = num_att_maps
        self.input_v_dim = input_v_dim
        self.input_q_dim = input_q_dim
        self.hidden_dim = hidden_dim
        self.num_out_heads = num_out_heads

        self.v_net = FCNet([input_v_dim, hidden_dim * self.num_att_maps], activation=activation, dropout=dropout)
        self.q_net = FCNet([input_q_dim, hidden_dim * self.num_att_maps], activation=activation, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < num_att_maps:
            self.p_net = nn.AvgPool1d(self.num_att_maps, stride=self.num_att_maps)

        if num_out_heads <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, num_out_heads, 1, hidden_dim * self.num_att_maps).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, num_out_heads, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(hidden_dim * self.num_att_maps, num_out_heads), dim=None)

        self.bn = nn.BatchNorm1d(hidden_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum("bvk,bvq,bqk->bk", (v, att_map, q))
        if 1 < self.num_att_maps:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.num_att_maps  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        num_v = v.size(1)
        num_q = q.size(1)
        if self.num_out_heads <= self.c:
            embed_v = self.v_net(v)
            embed_q = self.q_net(q)
            att_maps = torch.einsum("xhyk,bvk,bqk->bhvq", (self.h_mat, embed_v, embed_q)) + self.h_bias
        else:
            embed_v = self.v_net(v)
            embed_q = self.q_net(q)
            d_ = torch.matmul(
                embed_v.transpose(1, 2).unsqueeze(3), embed_q.transpose(1, 2).unsqueeze(2)
            )  # b x hidden_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x num_out_heads
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x num_out_heads x v x q
        if softmax:
            prob_mat = nn.functional.softmax(att_maps.view(-1, self.num_out_heads, num_v * num_q), 2)
            att_maps = prob_mat.view(-1, self.num_out_heads, num_v, num_q)
        logits = self.attention_pooling(embed_v, embed_q, att_maps[:, 0, :, :])
        for i in range(1, self.num_out_heads):
            logits_i = self.attention_pooling(embed_v, embed_q, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

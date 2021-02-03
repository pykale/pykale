# Created by Raivo Koot from modifying https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the NIPS2017 paper
    'Attention Is All You Need' about Transformers
    (https://arxiv.org/abs/1706.03762).
    Essentially, for all timesteps in a given sequence,
    adds information about the relative temporal location of a timestep
    directly into the features of that timestep, and then returns this
    slightly-modified, same-shape sequence.

    args:
        d_model: the number of features that each timestep has (required).
        max_len: the maximum sequence length that the positional
                  encodings should support (required).
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
        Expects input of shape (sequence_length, batch_size, num_features)
        and returns output of the same shape. sequence_length is at most
        allowed to be self.max_len and num_features is expected to
        be exactly self.d_model

        Args:
            x: a sequence input of shape (sequence_length, batch_size, num_features) (required).
        """
        x = x * self.scaling_term  # make embedding relatively larger than positional encoding
        x = x + self.pe[: x.size(0), :]
        return x

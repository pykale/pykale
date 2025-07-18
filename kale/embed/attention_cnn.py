from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from kale.embed.positional_encoding import PositionalEncoding
from kale.predict.class_domain_nets import FCNet
from kale.prepdata.tensor_reshape import seq_to_spatial, spatial_to_seq


class ContextCNNGeneric(nn.Module):
    """
    A template to construct a feature extractor consisting of a CNN followed by a
    sequence-to-sequence contextualizer like a Transformer-Encoder. Before inputting the CNN output
    tensor to the contextualizer, the tensor's spatial dimensions are unrolled
    into a sequence.

    Args:
        cnn (nn.Module): any convolutional neural network that takes in batches of images of
             shape (batch_size, channels, height, width) and outputs tensor
             representations of shape (batch_size, out_channels, out_height, out_width).
        cnn_output_shape (tuple): A tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN (required).
        contextualizer (nn.Module, optional): A sequence-to-sequence model that takes inputs of shape
                         (num_timesteps, batch_size, num_features) and uses
                         attention to contextualize the sequence and returns
                         a sequence of the exact same shape. This will mainly be
                         a Transformer-Encoder (required).
        output_type (string): One of 'sequence' or 'spatial'. If Spatial then the final
                      output of the model, which is a sequence, will be reshaped
                      to resemble the image-batch shape of the output of the CNN.
                      If Sequence then the output sequence is returned as is (required).

    Examples:
        >>> cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3),
        >>>                     nn.Conv2d(32, 64, kernel_size=3),
        >>>                     nn.MaxPool2d(2))
        >>> cnn_output_shape = (-1, 64, 8, 8)
        >>> contextualizer = nn.TransformerEncoderLayer(...)
        >>> output_type = 'spatial'
        >>>
        >>> attention_cnn = ContextCNNGeneric(cnn, cnn_output_shape, contextualizer, output_type)
        >>> output = attention_cnn(torch.randn((32,3,16,16)))
        >>>
        >>> output.size() == cnn_output_shape # True
    """

    def __init__(
        self,
        cnn: nn.Module,
        cnn_output_shape: Tuple[int, int, int, int],
        contextualizer: Union[nn.Module, Any],
        output_type: str,
    ):
        super(ContextCNNGeneric, self).__init__()
        assert output_type in ["spatial", "sequence"], (
            "parameter 'output_type' must be one of ('spatial', 'sequence')" + f" but is {output_type}"
        )

        self.cnn = cnn
        self.cnn_output_shape = cnn_output_shape
        self.contextualizer = contextualizer
        self.output_type = output_type

    def forward(self, x: torch.Tensor):
        """
        Pass the input through the cnn and then the contextualizer.

        Args:
            x: input image batch exactly as for CNNs (required).
        """
        cnn_rep = self.cnn(x)
        seq_rep = spatial_to_seq(cnn_rep)
        seq_rep = self.contextualizer(seq_rep)

        output = seq_rep
        if self.output_type == "spatial":
            desired_height = self.cnn_output_shape[2]
            desired_width = self.cnn_output_shape[3]
            output = seq_to_spatial(output, desired_height, desired_width)

        return output


class CNNTransformer(ContextCNNGeneric):
    """
    A feature extractor consisting of a given CNN backbone followed by a standard
    Transformer-Encoder. See documentation of "ContextCNNGeneric" for more
    information.

    Args:
        cnn (nn.Module): any convolutional neural network that takes in batches of images of
             shape (batch_size, channels, height, width) and outputs tensor
             representations of shape (batch_size, out_channels, out_height, out_width) (required).
        cnn_output_shape (tuple): a tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN (required).
        num_layers (int): number of attention layers in the Transformer-Encoder (required).
        num_heads (int): number of attention heads in each transformer block (required).
        dim_feedforward (int): number of neurons in the intermediate dense layer of
                          each transformer feedforward block (required).
        dropout (float): dropout rate of the transformer layers (required).
        output_type (string): one of 'sequence' or 'spatial'. If Spatial then the final
                      output of the model, which is the sequence output of the
                      Transformer-Encoder, will be reshaped to resemble the
                      image-batch shape of the output of the CNN (required).
        positional_encoder (nn.Module): None or a nn.Module that expects inputs of
                            shape (sequence_length, batch_size, embedding_dim)
                            and returns the same input after adding
                            some positional information to the embeddings. If
                            `None`, then the default and fixed sin-cos positional
                            encodings of base transformers are applied (optional).

    Examples:
        See pykale/examples/cifar_cnntransformer/model.py
    """

    def __init__(
        self,
        cnn: nn.Module,
        cnn_output_shape: Tuple[int, int, int, int],
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        output_type: str,
        positional_encoder: nn.Module = None,
    ):
        num_channels = cnn_output_shape[1]
        height = cnn_output_shape[2]
        width = cnn_output_shape[3]

        encoder_layer = nn.TransformerEncoderLayer(num_channels, num_heads, dim_feedforward, dropout)
        encoder_normalizer = nn.LayerNorm(num_channels)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_normalizer)

        if positional_encoder is None:
            positional_encoder = PositionalEncoding(d_model=num_channels, max_len=height * width)
        else:
            # allows for passing the identity block to skip this step
            # or chosing a different encoding
            positional_encoder = positional_encoder

        transformer_input_dropout = nn.Dropout(dropout)
        contextualizer = nn.Sequential(positional_encoder, transformer_input_dropout, encoder)

        super(CNNTransformer, self).__init__(cnn, cnn_output_shape, contextualizer, output_type)

        # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class BANLayer(nn.Module):
    """
    The bilinear Attention Network (BAN) layer is designed to apply bilinear attention between two feature sets (`v` and `q`),
    which could represent features extracted from drugs and proteins, respectively. This layer
    enables the interaction between these two sets of features, allowing the model to learn
    joint representations that can be used for downstream tasks like predicting drug-protein
    interactions.

    Args:
        input_v_dim (int): Dimensionality of the first input feature set (`v`).
        input_q_dim (int): Dimensionality of the second input feature set (`q`).
        hidden_dim (int): Dimensionality of the hidden layer used in the bilinear attention mechanism.
        num_out_heads (int): Number of output heads in the bilinear attention mechanism.
        activation (str, optional): Activation function to use in the fully connected networks for `v` and `q`.
                             Default is "ReLU".
        dropout (float, optional): Dropout rate to apply after each layer in the fully connected networks.
                                   Default is 0.2.
        num_att_maps (int, optional): Number of attention maps to generate (used in pooling). Default is 3.
    """

    def __init__(
        self, input_v_dim, input_q_dim, hidden_dim, num_out_heads, activation="ReLU", dropout=0.2, num_att_maps: int = 3
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
            self.h_net = weight_norm(nn.Linear(hidden_dim * self.num_att_maps, num_out_heads), dim=0)

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
            embed_v = self.v_net(v).transpose(1, 2).unsqueeze(3)
            embed_q = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(embed_v, embed_q)  # b x hidden_dim x v x q
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

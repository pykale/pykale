import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.embed.positional_encoding import PositionalEncoding
from kale.embed.video_selayer import SELayerFeat
from kale.prepdata.tensor_reshape import seq_to_spatial, spatial_to_seq


class ContextCNNGeneric(nn.Module):
    """
    A template to construct a feature extractor consisting of a CNN followed by a
    sequence-to-sequence contextualizer like a Transformer-Encoder. Before inputting the CNN output
    tensor to the contextualizer, the tensor's spatial dimensions are unrolled
    into a sequence.

    Args:
        cnn: any convolutional neural network that takes in batches of images of
             shape (batch_size, channels, height, width) and outputs tensor
             representations of shape (batch_size, out_channels, out_height, out_width).
        cnn_output_shape: A tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN (required).
        contextualizer: A sequence-to-sequence model that takes inputs of shape
                         (num_timesteps, batch_size, num_features) and uses
                         attention to contextualize the sequence and returns
                         a sequence of the exact same shape. This will mainly be
                         a Transformer-Encoder (required).
        output_type: One of 'sequence' or 'spatial'. If Spatial then the final
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
        self, cnn: nn.Module, cnn_output_shape: Tuple[int, int, int, int], contextualizer: nn.Module, output_type: str
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
        cnn: any convolutional neural network that takes in batches of images of
             shape (batch_size, channels, height, width) and outputs tensor
             representations of shape (batch_size, out_channels, out_height, out_width) (required).
        cnn_output_shape: a tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN (required).
        num_layers: number of attention layers in the Transformer-Encoder (required).
        num_heads: number of attention heads in each transformer block (required).
        dim_feedforward: number of neurons in the intermediate dense layer of
                          each transformer feedforward block (required).
        dropout: dropout rate of the transformer layers (required).
        output_type: one of 'sequence' or 'spatial'. If Spatial then the final
                      output of the model, which is the sequence output of the
                      Transformer-Encoder, will be reshaped to resemble the
                      image-batch shape of the output of the CNN (required).
        positional_encoder: None or a nn.Module that expects inputs of
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


class SelfAttention(nn.Module):
    """
    A vanilla multi-head attention layer with a projection at the end. Can be set to causal or not causal.
    """

    def __init__(
        self, emb_dim, num_heads, att_dropout, final_dropout, causal=False, max_seq_len=10000, use_performer_att=False
    ):
        super().__init__()
        assert emb_dim % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        # regularization
        self.att_dropout = nn.Dropout(att_dropout)
        self.final_dropout = nn.Dropout(final_dropout)
        # output projection
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.causal = causal
        if causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
            )

        self.num_heads = num_heads

        self.use_performer_att = use_performer_att
        # if self.use_performer_att:
        #     self.performer_att = FastAttention(dim_heads=emb_dim//num_heads, nb_features=emb_dim, causal=False)

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        if not self.use_performer_att:
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.att_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            y = self.performer_att(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.final_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """
    Standard transformer block consisting of multi-head attention and two-layer MLP.
    """

    def __init__(
        self, emb_dim, num_heads, att_dropout, att_resid_dropout, final_dropout, max_seq_len, ff_dim, causal=False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = SelfAttention(emb_dim, num_heads, att_dropout, att_resid_dropout, causal, max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, emb_dim), nn.Dropout(final_dropout),
        )

    def forward(self, x):
        # BATCH, TIME, CHANNELS = x.size()
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerSENet(nn.Module):
    """Regular simple network for video input.
    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        output_size (int, optional): the dimension of output.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
    """

    def __init__(self, input_size=512, n_channel=512, output_size=256, dropout_keep_prob=0.5):
        super(TransformerSENet, self).__init__()
        self.hidden_sizes = 512
        self.num_layers = 4

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=input_size,
                    num_heads=8,
                    att_dropout=0.1,
                    att_resid_dropout=0.1,
                    final_dropout=0.1,
                    max_seq_len=9,
                    ff_dim=self.hidden_sizes,
                    causal=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.fc1 = nn.Linear(input_size, n_channel)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_keep_prob)
        self.fc2 = nn.Linear(n_channel, output_size)
        self.selayer = SELayerFeat(channel=8, reduction=2)

    def forward(self, x):
        for layer in self.transformer:
            x = layer(x)
        x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
        x = self.selayer(x)
        return x

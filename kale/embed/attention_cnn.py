import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.prepdata.tensor_reshape import spatial_to_seq, \
                                         seq_to_spatial
from kale.embed.positional_encoding import PositionalEncoding


class ContextCNNGeneric(nn.Module):
    """
    A template to construct a feature extractor consisting of a CNN followed by a
    sequence-to-sequence contextualizer like a Transformer. Before inputting the CNN output
    tensor to the contextualizer, the tensor's spatial dimensions are unrolled
    into a sequence.

    args:
        cnn - A convolutional neural network that takes in images and
              outputs tensors.
        cnn_output_shape - A tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN.
        contextualizer - A sequence-to-sequence model that takes inputs of shape
                         (num_timesteps, batch_size, num_features) and uses
                         attention to contextualize the sequence and returns
                         a sequence of the exact same shape. This will mainly be
                         a Transformer-Encoder.
        output_type - One of 'sequence' or 'spatial'. If Spatial then the final
                      output of the model, which is a sequence, will be reshaped
                      to resemble the image-batch shape of the output of the CNN.
                      If Sequence then the output sequence is returned as is.
    """

    def __init__(self, cnn, cnn_output_shape, contextualizer, output_type):
        super(ContextCNNGeneric, self).__init__()
        assert output_type in ['spatial', 'sequence'], \
            "parameter 'output_type' must be one of ('spatial', 'sequence')" +\
            f" but is {output_type}"

        self.cnn = cnn
        self.cnn_output_shape = cnn_output_shape
        self.contextualizer = contextualizer
        self.output_type = output_type

    def forward(self, x):
        cnn_rep = self.cnn(x)
        seq_rep = spatial_to_seq(cnn_rep)
        seq_rep = self.contextualizer(seq_rep)

        output = seq_rep
        if self.output_type == 'spatial':
            desired_height = self.cnn_output_shape[2]
            desired_width = self.cnn_output_shape[3]
            output = seq_to_spatial(output, desired_height, desired_width)

        return output

class CNNTransformer(ContextCNNGeneric):
    """
    A feature extractor consisting of a CNN backbone followed by a standard
    Transformer-Encoder.

    args:
        cnn - a convolutional neural network that takes in images and
              outputs tensors.
        cnn_output_shape - a tuple of shape (batch_size, num_channels, height, width)
                           describing the output shape of the given CNN.
        num_layers - number of attention layers in the Transformer-Encoder.
        num_heads - number of attention heads in each transformer block.
        dim_feedforward - number of neurons in the intermediate dense layer of
                          each transformer feedforward block.
        dropout - dropout rate of the transformer layers.
        output_type - one of 'sequence' or 'spatial'. If Spatial then the final
                      output of the model, which is the sequence output of the
                      Transformer-Encoder, will be reshaped to resemble the
                      image-batch shape of the output of the CNN.
    """

    def __init__(self, cnn, cnn_output_shape, num_layers, num_heads, dim_feedforward,
                dropout, output_type):

        num_channels = cnn_output_shape[1]
        height = cnn_output_shape[2]
        width = cnn_output_shape[3]

        encoder_layer = nn.TransformerEncoderLayer(num_channels, num_heads, \
                                                   dim_feedforward, dropout)
        encoder_normalizer = nn.LayerNorm(num_channels)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_normalizer)

        positional_encoder = PositionalEncoding(d_model=num_channels, max_len=height*width)
        transformer_input_dropout = nn.Dropout(dropout)
        contextualizer = nn.Sequential(positional_encoder, transformer_input_dropout, encoder)

        super(CNNTransformer, self).__init__(cnn, cnn_output_shape, \
                                             contextualizer, output_type)

        # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        # source code
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

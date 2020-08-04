from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..embed.attention_cnn import ContextCNNGeneric
from ..embed.positional_encoding import PositionalEncoding
from ..embed.linformer import LinearTransformerEncoderLayer


class VideoTransformer(pl.LightningModule):
    """
    A feature extractor for videos consisting of three consecutive modules.\n
    1. A CNN without a head that is applied independently on each frame of the
    video.\n
    2. A Transformer-Encoder that is applied independently on the output CNN-representation
    of each frame after unrolling their remaining spatial dimensions into a sequence.\n
    3. A Transformer-Encoder that is applied after concatenating each frame's sequence
    into one big sequence. \n\n
    
    So, first the VideoTransformer extracts per-frame features through a CNN, then 
    per-frame contextualizes these features with self-attention, and finally globally
    contextualizes these features, as a whole, with self-attention.

    Note:
        This module is a video-to-sequence model where a custom head
        can be used ontop to customize this model for different types of tasks (or just
        classification), in the same way BERT's output sequences can be used for 
        various tasks.

    Args:
        fram_per_vid: the number of frames each input video has. This must always
                      be the same during training and inference (required).
        cnntransformer: a feature extractor built from kale.embed.attention_cnn
                        consisting of a cnn with a transformer-encoder stacked ontop 
                        to process images (required).
        cnntrans_out_shape: the shape that the cnntransformer will output in 
                            the format (seq_len, batch_size, channels) (required).
        nheads: number of attention heads in the `step 3` transformer-encoder (required).
        linformer_k: number of down projection dimensions of the `step 3`
                     transformer-encoder. See `Linformer` https://arxiv.org/abs/2006.04768 
                     for more details (required).
        dim_feedforward: number of neurons in the intermediate dense layer of
                         each `step 3` Transformer-Encoder feedforward block (required). 
        num_layers: number of attention layers in the `step 3` Transformer-Encoder (required).
        dropout: dropout rate of the `step 3` Transformer-Encoder layers (default=0.1).
        activation: 'relu' or 'gelu' for the `step 3` Transformer-Encoder (default='relu).
        pos_encoding: None, a custom positional-encoding block, or an identity block
                      that is applied on the final long sequences before `step 3`. If
                      None, the default sin-cos encodings will be applied (default=None).


    Examples:
        >>> fram_per_vid = 8
        >>> channels = 3
        >>> input_shape = (-1, fram_per_vid, channels, 16, 16)
        >>>
        >>> cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3), 
        >>>                     nn.Conv2d(32, 64, kernel_size=3),
        >>>                     nn.MaxPool2d(2))
        >>> cnn_output_shape = (-1, 64, 8, 8)
        >>>
        >>> contextualizer = nn.TransformerEncoderLayer(d_model=64, ...)
        >>> output_type = 'sequence'
        >>> cnntransformer = ContextCNNGeneric(cnn, cnn_output_shape, contextualizer, output_type)
        >>> cnntrans_out_shape = (8*8, -1, 64)
        >>>
        >>> model = VideoTransformer(fram_per_vid, cnntransformer, cnntrans_out_shape, ...)
        >>>
        >>> batch_size = 4
        >>> input = torch.randn((batch_size, fram_per_vid, channels, 16, 16))
        >>>
        >>> model(input).size() == (fram_per_vid*8*8, batch_size, 64)  # True
    """

    def __init__(self, fram_per_vid: int, cnntransformer: ContextCNNGeneric, 
                cnntrans_out_shape: Tuple[int, int, int],
                nheads: int, linformer_k: int, dim_feedforward: int, num_layers: int,
                dropout: float=0.1, activation: str='relu', pos_encoding: nn.Module=None):
        super(VideoTransformer, self).__init__()

        self.cnntransformer = cnntransformer
        self.fram_per_vid = fram_per_vid
        
        out_channels = cnntrans_out_shape[2]
        final_seq_len = cnntrans_out_shape[0] * fram_per_vid
        encoder_layer = LinearTransformerEncoderLayer(d_model=out_channels, 
                                                      nhead=nheads, 
                                                      seq_len=final_seq_len, 
                                                      proj_k=linformer_k,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.linformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        if pos_encoding is None:
            pos_encoding = PositionalEncoding(d_model=out_channels, max_len=final_seq_len)
        self.pos_enc = pos_encoding

        # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        for p in self.linformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Applies the cnntransformer independently on each frame of
        each video and then concatenates the resulting sequence representations
        of each video's frames together into one long sequence per video
        and applies a linformer-encoder to further contextualize each video
        sequence. 

        Args:
            x: a batch of videos with shape 
               (batch_size, fram_per_vid, channels, height, width)

        Output Shape:
            Returns output with shape (batch_size, final_seq_len, channels) where
            the final sequence length will be out_height*out_width*fram_per_video
            where out_height and out_width are the remaining spatial dimensions of the
            CNNs output. Channels is the number of channels the CNN output has.
        """
        batch_size = x.size(0)

        # Turn batch and frame dim into a single longer batch of images.
        # This way cnntransformer can process all images of all videos
        # in the batch in parallel
        x = torch.flatten(x, end_dim=1)
        x = self.cnntransformer(x)

        seq_len = x.size(0)
        num_dim = x.size(2)
        
        # put the Batch dimension in front
        x = x.permute((1,0,2)) 
        # fold batch back into batch and frames
        x = x.view((batch_size, self.fram_per_vid, seq_len, num_dim))
        x = torch.flatten(x.permute((1,2,0,3)), end_dim=1) # flatten frame and seq dimension
                                                           # into one long seq

        x = self.pos_enc(x)
        x = self.linformer(x)
        return x.permute(1,0,2) # put the batch dimension first
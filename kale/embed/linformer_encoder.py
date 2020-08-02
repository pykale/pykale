# Copy-paste with slight modification from torch.nn.TransformerEncoderLayer

import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from linformer_attention import LinearMultiheadAttention
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

class LinearTransformerEncoderLayer(Module):
    r"""Modification of PyTorch's nn.TransformerEncoderLayer.

    This modification reduces the computational cost of the self-attention module from
    O(n^2) to O(n) by implementing the proposed adjusted linear attention block from:
    `Linformer: Self-Attention with Linear Complexity` (2020) (https://arxiv.org/abs/2006.04768).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        seq_len: the sequence length (required).
        proj_k: the projected dimension `k` of key and value (default=128).
        param_sharing: parameter sharing mode: layerwise, none. headwise is not implemented (default='none').
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> # project a sequence length of 30,000 to a sequence length of 512 and back to 30000
        >>> encoder_layer = LinearTransformerEncoderLayer(d_model=128, nhead=8, seq_len=30000, proj_k=512)
        >>>
        >>> src = torch.rand(30000, 32, 128)
        >>> out = encoder_layer(src)
        >>> out.size() == (30000, 32, 128) # True

    """

    def __init__(self, d_model: int, nhead: int, 
                seq_len: int, proj_k: int=128, proj_param_sharing: str='none',
                dim_feedforward: int=2048, dropout: float=0.1,
                activation: str="relu"):
        super(LinearTransformerEncoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = LinearMultiheadAttention(d_model, nhead, dropout=dropout, seq_len=seq_len,
                                                 proj_k=proj_k, param_sharing=proj_param_sharing)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LinearTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in PyTorch Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
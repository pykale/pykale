import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from performer_pytorch import FastAttention
from kale.embed.video_selayer import SELayerFeat


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

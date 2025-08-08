# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
#         Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Xianyuan Liu, xianyuan.liu@sheffield.ac.uk
# =============================================================================

"""
Python implementation of DrugBAN model for predicting drug-protein interactions using PyTorch.

This includes a Graph Convolutional Network (GCN) for extracting features from molecular graphs (drugs),
and a Convolutional Neural Network (CNN) for extracting features from protein sequences.

These features are fused using a Bilinear Attention Network (BAN) layer and passed through an MLP classifier
to predict interactions.
"""


from dataclasses import dataclass

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from kale.embed.attention import BANLayer
from kale.embed.cnn import ProteinCNN
from kale.embed.gcn import MolecularGCN
from kale.predict.decode import MLPDecoder


@dataclass
class DrugConfig:
    """
    Configuration for drug graph input processing.

    Attributes:
        NODE_IN_FEATS (int): Dimension of the raw input node features.
        NODE_IN_EMBEDDING (int): Dimension of the node embeddings after initial transformation.
        HIDDEN_LAYERS (list): List of hidden layer sizes for drug feature extraction.
        PADDING (bool): Whether to apply padding to drug graph inputs.

    Reference:
        Bai, P., Miljković, F., John, B., & Lu, H. (2023). Interpretable bilinear attention network with domain
        adaptation improves drug–target prediction. Nature Machine Intelligence, 5(2), 126-136.
        For initial implementation, please go to https://github.com/peizhenbai/DrugBAN
    """

    NODE_IN_FEATS: int
    NODE_IN_EMBEDDING: int
    HIDDEN_LAYERS: list
    PADDING: bool


@dataclass
class ProteinConfig:
    """
    Configuration for protein sequence input processing.

    Attributes:
        EMBEDDING_DIM (int): Dimension of the protein embedding vector.
        NUM_FILTERS (list): List specifying the number of filters in each convolutional layer.
        KERNEL_SIZE (list): List specifying the kernel size for each convolutional layer.
        PADDING (bool): Whether to apply padding to the protein input sequence.
    """

    EMBEDDING_DIM: int
    NUM_FILTERS: list  # Number of filters in each convolutional layer
    KERNEL_SIZE: list  # Kernel size for each convolutional layer
    PADDING: bool


@dataclass
class DecoderConfig:
    """
    Configuration for the MLP decoder that predicts drug-protein interactions.

    Attributes:
        IN_DIM (int): Input dimension to the decoder.
        HIDDEN_DIM (int): Dimension of the decoder's hidden layer(s).
        OUT_DIM (int): Dimension of the decoder's output before final classification.
        BINARY (int): Number of output classes. Use 1 for binary classification.
    """

    IN_DIM: int
    HIDDEN_DIM: int
    OUT_DIM: int
    BINARY: int


@dataclass
class BCNConfig:
    """
    Configuration for the Bilinear Attention Network (BCN) component.

    Attributes:
        HEADS (int): Number of attention heads in the BCN module.
    """

    HEADS: int


@dataclass
class FullConfig:
    """
    Aggregated configuration for the full model, combining sub-configs for
    drug input, protein input, decoder, and BCN components.

    Attributes:
        DRUG (DrugConfig): Configuration for drug graph features.
        PROTEIN (ProteinConfig): Configuration for protein sequence features.
        DECODER (DecoderConfig): Configuration for the MLP decoder.
        BCN (BCNConfig): Configuration for the Bilinear Attention Network.
    """

    DRUG: DrugConfig
    PROTEIN: ProteinConfig
    DECODER: DecoderConfig
    BCN: BCNConfig


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

    def __init__(self, config: FullConfig):
        super(DrugBAN, self).__init__()
        # Drug config
        drug_in_feats = config.DRUG.NODE_IN_FEATS
        drug_embedding = config.DRUG.NODE_IN_EMBEDDING
        drug_hidden_feats = config.DRUG.HIDDEN_LAYERS
        drug_padding = config.DRUG.PADDING

        # Protein config
        protein_emb_dim = config.PROTEIN.EMBEDDING_DIM
        num_filters = config.PROTEIN.NUM_FILTERS
        kernel_size = config.PROTEIN.KERNEL_SIZE
        protein_padding = config.PROTEIN.PADDING

        # Decoder config
        mlp_in_dim = config.DECODER.IN_DIM
        mlp_hidden_dim = config.DECODER.HIDDEN_DIM
        mlp_out_dim = config.DECODER.OUT_DIM
        out_binary = config.DECODER.BINARY

        # BCN config
        ban_heads = config.BCN.HEADS

        self.molecular_extractor = MolecularGCN(
            in_feats=drug_in_feats, dim_embedding=drug_embedding, padding=drug_padding, hidden_feats=drug_hidden_feats
        )
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(
                input_v_dim=drug_hidden_feats[-1],
                input_q_dim=num_filters[-1],
                hidden_dim=mlp_in_dim,
                num_out_heads=ban_heads,
            ),
            name="h_mat",
            dim=0,
        )
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, input_molecular, input_protein, mode="train"):
        feat_molecular = self.molecular_extractor(input_molecular)
        feat_protein = self.protein_extractor(input_protein)
        f, att = self.bcn(feat_molecular, feat_protein)
        score = self.mlp_classifier(f)
        if mode == "train":
            return feat_molecular, feat_protein, f, score
        elif mode == "eval":
            return feat_molecular, feat_protein, f, score, att
        else:
            # Optionally raise error on unexpected mode
            raise ValueError(f"Unsupported mode: {mode}")

# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
Define the learning model and configure model parameters.
Constructed from https://github.com/pliang279/MultiBench/tree/main/examples/multimedia
"""

from kale.embed.lenet import LeNet
from kale.embed.multimodal_common_fusions import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
from kale.pipeline.mmdl import MMDL
from kale.predict.two_layered_mlp import MLP


def get_model(cfg, device):
    """
    :encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation

    Builds and returns a MMDL model.

    Args:
        cfg: A YACS config object.
    """
    encoders = [
        LeNet(cfg.MODEL.LENET_IN_CHANNELS, cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_IMG),
        LeNet(cfg.MODEL.LENET_IN_CHANNELS, cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_AUD),
    ]

    if cfg.MODEL.FUSION == "late":
        fusion = Concat()
        head = MLP(cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)
    elif cfg.MODEL.FUSION == "tesnor_matrix":
        fusion = MultiplicativeInteractions2Modal(
            cfg.MODEL.MULTIPLICATIVE_FUSION_IN_DIM,
            cfg.MODEL.MULTIPLICATIVE_FUSION_OUT_DIM,
            cfg.MODEL.MULTIPLICATIVE_OUTPUT,
        )
        head = MLP(cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)
    elif cfg.MODEL.FUSION == "low_rank_tensor":
        fusion = LowRankTensorFusion(
            cfg.MODEL.LOW_RANK_TENSOR_IN_DIM, cfg.MODEL.LOW_RANK_TENSOR_OUT_DIM, cfg.MODEL.LOW_RANK_TENSOR_RANK
        )
        head = MLP(cfg.MODEL.MLP_LOW_RANK_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)

    model = MMDL(encoders, fusion, head, has_padding=cfg.SOLVER.IS_PACKED).to(device)
    print("Model loaded succesfully")

    return model

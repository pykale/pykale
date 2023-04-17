# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
Define the learning model and configure model parameters.
References: 1. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_simple_late_fusion.py
            2. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_low_rank_tensor.py
            3. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_multi_interac_matrix.py
"""

from kale.embed.lenet import LeNet
from kale.embed.multimodal_fusion_methods import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
from kale.pipeline.multi_modal_module import MultiModalDeepLearning
from kale.predict.mlp_classifier import MLPClassifier


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
        head = MLPClassifier(cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)
    elif cfg.MODEL.FUSION == "tesnor_matrix":
        fusion = MultiplicativeInteractions2Modal(
            cfg.MODEL.MULTIPLICATIVE_FUSION_IN_DIM,
            cfg.MODEL.MULTIPLICATIVE_FUSION_OUT_DIM,
            cfg.MODEL.MULTIPLICATIVE_OUTPUT,
        )
        head = MLPClassifier(cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)
    elif cfg.MODEL.FUSION == "low_rank_tensor":
        fusion = LowRankTensorFusion(
            cfg.MODEL.LOW_RANK_TENSOR_IN_DIM, cfg.MODEL.LOW_RANK_TENSOR_OUT_DIM, cfg.MODEL.LOW_RANK_TENSOR_RANK
        )
        head = MLPClassifier(cfg.MODEL.MLP_LOW_RANK_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)

    model = MultiModalDeepLearning(encoders, fusion, head, has_padding=cfg.SOLVER.IS_PACKED).to(device)
    print("Model loaded succesfully")

    return model

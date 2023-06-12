"""
Define the learning model and configure model hyperparameters.
References: 1. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_simple_late_fusion.py
            2. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_low_rank_tensor.py
            3. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_multi_interac_matrix.py
"""

from kale.embed.feature_fusion import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
from kale.embed.image_cnn import LeNet
from kale.pipeline.base_nn_trainer import MultiModalTrainer
from kale.pipeline.multimodal_deep_learning import MultiModalDeepLearning
from kale.predict.decode import MLPDecoder


def get_model(cfg, device):
    """
    Builds and returns an MMDL model.

    Args:
        cfg: A YACS config object.
    """
    encoders = [
        LeNet(cfg.MODEL.LENET_IN_CHANNELS, cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_IMG),
        LeNet(cfg.MODEL.LENET_IN_CHANNELS, cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_AUD),
    ]

    if cfg.MODEL.FUSION == "late":
        fusion = Concat()
        head = MLPDecoder(
            cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_additional_layers=False
        )
    elif cfg.MODEL.FUSION == "tensor_matrix":
        fusion = MultiplicativeInteractions2Modal(
            cfg.MODEL.MULTIPLICATIVE_FUSION_IN_DIM,
            cfg.MODEL.MULTIPLICATIVE_FUSION_OUT_DIM,
            cfg.MODEL.MULTIPLICATIVE_OUTPUT,
        )
        head = MLPDecoder(
            cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_additional_layers=False
        )
    elif cfg.MODEL.FUSION == "low_rank_tensor":
        fusion = LowRankTensorFusion(
            cfg.MODEL.LOW_RANK_TENSOR_IN_DIM, cfg.MODEL.LOW_RANK_TENSOR_OUT_DIM, cfg.MODEL.LOW_RANK_TENSOR_RANK
        )
        head = MLPDecoder(
            cfg.MODEL.MLP_LOW_RANK_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_additional_layers=False
        )

    classifier = MultiModalDeepLearning(encoders, fusion, head).to(device)

    model = MultiModalTrainer(classifier, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return model

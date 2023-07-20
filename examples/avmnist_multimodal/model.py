"""
Define and build the Multimodal Neural Network (MMNN) model based on chosen hyperparameters..
References: 1. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_simple_late_fusion.py
            2. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_low_rank_tensor.py
            3. https://github.com/pliang279/MultiBench/blob/main/examples/multimedia/avmnist_multi_interac_matrix.py
"""

from kale.embed.feature_fusion import BimodalInteractionFusion, Concat, LowRankTensorFusion
from kale.embed.image_cnn import LeNet
from kale.pipeline.base_nn_trainer import MultimodalNNTrainer
from kale.predict.decode import MLPDecoder


def get_model(cfg, device):
    """
    Builds and returns an MMNN model according to the config object passed.

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
            cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_decoder_layers=False
        )
    elif cfg.MODEL.FUSION == "bimodal_interaction_fusion":
        fusion = BimodalInteractionFusion(
            cfg.MODEL.MULTIPLICATIVE_FUSION_IN_DIM,
            cfg.MODEL.MULTIPLICATIVE_FUSION_OUT_DIM,
            cfg.MODEL.MULTIPLICATIVE_OUTPUT,
        )
        head = MLPDecoder(
            cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_decoder_layers=False
        )
    elif cfg.MODEL.FUSION == "low_rank_tensor":
        fusion = LowRankTensorFusion(
            cfg.MODEL.LOW_RANK_TENSOR_IN_DIM, cfg.MODEL.LOW_RANK_TENSOR_OUT_DIM, cfg.MODEL.LOW_RANK_TENSOR_RANK
        )
        head = MLPDecoder(
            cfg.MODEL.MLP_LOW_RANK_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM, include_decoder_layers=False
        )

    model = MultimodalNNTrainer(encoders, fusion, head, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return model

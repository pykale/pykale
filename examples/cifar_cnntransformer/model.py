"""
Define and build the model based on chosen hyperparameters.
"""
import torch
import torch.nn as nn
from trainer import CNNTransformerTrainer

from kale.embed.attention_cnn import CNNTransformer, ContextCNNGeneric
from kale.embed.image_cnn import SimpleCNN
from kale.predict.class_domain_nets import ClassNet


def get_model(cfg):
    """
    Builds and returns a model according to the config
    object passed.

    Args:
        cfg: A YACS config object.
    """

    cnn = SimpleCNN(cfg.CNN.CONV_LAYERS, cfg.CNN.ACTIVATION_FUN, cfg.CNN.USE_BATCHNORM, cfg.CNN.POOL_LOCATIONS)

    if cfg.TRANSFORMER.USE_TRANSFORMER:
        context_cnn = CNNTransformer(
            cnn,
            cfg.CNN.OUTPUT_SHAPE,
            cfg.TRANSFORMER.NUM_LAYERS,
            cfg.TRANSFORMER.NUM_HEADS,
            cfg.TRANSFORMER.DIM_FEEDFORWARD,
            cfg.TRANSFORMER.DROPOUT,
            cfg.TRANSFORMER.OUTPUT_TYPE,
        )
    else:
        context_cnn = ContextCNNGeneric(
            cnn, cfg.CNN.OUTPUT_SHAPE, contextualizer=lambda x: x, output_type=cfg.TRANSFORMER.OUTPUT_TYPE
        )

    classifier = ClassNet(cfg.DATASET.NUM_CLASSES, cfg.CNN.OUTPUT_SHAPE)
    net = nn.Sequential(context_cnn, classifier)
    optim = torch.optim.SGD(
        net.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    model = CNNTransformerTrainer(model=net, optim=optim.state_dict(), cfg=cfg)
    return model, optim

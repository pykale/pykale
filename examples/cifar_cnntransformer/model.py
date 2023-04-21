"""
Define and build the model based on chosen hyperparameters.
"""
from copy import deepcopy

import torch
import torch.nn as nn

from kale.embed.attention_cnn import CNNTransformer, ContextCNNGeneric
from kale.embed.image_cnn import SimpleCNN
from kale.pipeline.base_trainer import CNNTransformerTrainer
from kale.predict.class_domain_nets import ClassNet


def get_config(cfg):
    """
    Sets the hyperparameter for the optimizer and experiment using the config file
    Args:
        cfg: A YACS config object.
    """

    config_params = {
        "train_params": {
            "init_lr": cfg.SOLVER.BASE_LR,
            "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            "lr_gamma": cfg.SOLVER.LR_GAMMA,
            "train_batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
            "test_batch_size": cfg.SOLVER.TEST_BATCH_SIZE,
            "max_epochs": cfg.SOLVER.MAX_EPOCHS,
            "warmup": cfg.SOLVER.WARMUP,
            "warmup_epochs": cfg.SOLVER.WARMUP_EPOCHS,
            "optimizer": {
                "type": cfg.SOLVER.TYPE,
                "optim_params": {"momentum": cfg.SOLVER.MOMENTUM, "weight_decay": cfg.SOLVER.WEIGHT_DECAY,},
            },
        },
        "data_params": {"num_classes": cfg.DATASET.NUM_CLASSES,},
        "cnn_params": {
            "conv_layers_spec": cfg.CNN.CONV_LAYERS,
            "activation_fun": cfg.CNN.ACTIVATION_FUN,
            "use_batchnorm": cfg.CNN.USE_BATCHNORM,
            "pool_locations": cfg.CNN.POOL_LOCATIONS,
        },
        "transformer_params": {
            "cnn_output_shape": cfg.CNN.OUTPUT_SHAPE,
            "num_layers": cfg.TRANSFORMER.NUM_LAYERS,
            "num_heads": cfg.TRANSFORMER.NUM_HEADS,
            "dim_feedforward": cfg.TRANSFORMER.DIM_FEEDFORWARD,
            "dropout": cfg.TRANSFORMER.DROPOUT,
            "output_type": cfg.TRANSFORMER.OUTPUT_TYPE,
        },
    }
    return config_params


def get_model(cfg):
    """
    Builds and returns a model according to the config object passed.

    Args:
        cfg: A YACS config object.
    """

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    data_params = config_params["data_params"]
    data_params_local = deepcopy(data_params)
    cnn_params = config_params["cnn_params"]
    cnn_params_local = deepcopy(cnn_params)
    transformer_params = config_params["transformer_params"]
    transformer_params_local = deepcopy(transformer_params)

    cnn = SimpleCNN(**cnn_params_local)

    if cfg.TRANSFORMER.USE_TRANSFORMER:
        context_cnn = CNNTransformer(cnn, **transformer_params_local)
    else:
        context_cnn = ContextCNNGeneric(
            cnn,
            cnn_params_local["cnn_output_shape"],
            contextualizer=lambda x: x,
            output_type=transformer_params_local["output_type"],
        )

    classifier = ClassNet(data_params_local["num_classes"], transformer_params_local["cnn_output_shape"])
    # net = nn.Sequential(context_cnn, classifier)
    # optim = torch.optim.SGD(
    #     net.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    # )
    # model = CNNTransformerTrainer(model=net, optimizer=optim.state_dict(), cfg=cfg)
    model = CNNTransformerTrainer(feature_extractor=context_cnn, task_classifier=classifier, **train_params_local)
    return model

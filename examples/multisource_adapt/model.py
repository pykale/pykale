"""
Define the learning model and configure training parameters.
"""
# Author: Shuo Zhou
# Initial Date: 09.09.2021

from copy import deepcopy

import torch

from kale.embed.image_cnn import ResNet18Feature, SmallCNNFeature
from kale.pipeline.multi_domain_adapter import create_ms_adapt_trainer
from kale.predict.class_domain_nets import ClassNetSmallImage


def get_config(cfg):
    """
    Sets the hyper-parameters for the optimizer and experiment using the config file

    Args:
        cfg: A YACS config object.
    """
    config_params = {
        "train_params": {
            "adapt_lambda": cfg.SOLVER.AD_LAMBDA,
            "adapt_lr": cfg.SOLVER.AD_LR,
            "lambda_init": cfg.SOLVER.INIT_LAMBDA,
            "nb_adapt_epochs": cfg.SOLVER.MAX_EPOCHS,
            "nb_init_epochs": cfg.SOLVER.MIN_EPOCHS,
            "init_lr": cfg.SOLVER.BASE_LR,
            "batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
            "optimizer": {
                "type": cfg.SOLVER.TYPE,
                "optim_params": {
                    "momentum": cfg.SOLVER.MOMENTUM,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                    "nesterov": cfg.SOLVER.NESTEROV,
                },
            },
        },
        "data_params": {
            "dataset_group": cfg.DATASET.NAME,
            "dataset_name": cfg.DATASET.NAME + "_Target_" + cfg.DATASET.TARGET,
            "source": "_".join(cfg.DATASET.SOURCE) if cfg.DATASET.SOURCE is not None else None,
            "target": cfg.DATASET.TARGET,
            "size_type": cfg.DATASET.SIZE_TYPE,
            "weight_type": cfg.DATASET.WEIGHT_TYPE,
        },
    }
    return config_params


# Based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
def get_model(cfg, dataset, num_channels):
    """
    Builds and returns a model and associated hyper-parameters according to the config object passed.

    Args:
        cfg: A YACS config object.
        dataset: A multidomain dataset consisting of source and target datasets.
        num_channels: The number of image channels.
    """
    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)

    # setup feature extractor
    if cfg.DATASET.NAME.upper() == "DIGITS":
        feature_network = SmallCNNFeature(num_channels)
    else:
        feature_network = ResNet18Feature(num_channels)

    if cfg.DAN.METHOD == "MFSAN":
        feature_network = torch.nn.Sequential(*(list(feature_network.children())[:-1]))

    method_params = {"n_classes": cfg.DATASET.NUM_CLASSES, "target_domain": cfg.DATASET.TARGET}

    model = create_ms_adapt_trainer(
        method=cfg.DAN.METHOD,
        dataset=dataset,
        feature_extractor=feature_network,
        task_classifier=ClassNetSmallImage,
        **method_params,
        **train_params_local,
    )

    return model, train_params

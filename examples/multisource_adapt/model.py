"""
Define the learning model and configure training parameters.
"""
# Author: Shuo Zhou
# Initial Date: 09.09.2021

from copy import deepcopy

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from kale.pipeline.domain_adapter import Method
from kale.pipeline.multi_domain_adapter import create_ms_adapt_trainer

# from kale.embed.image_cnn import SmallCNNFeature
from kale.embed.image_cnn import ResNet50Feature
from kale.predict.class_domain_nets import ClassNetSmallImage


def get_config(cfg):
    """
    Sets the hypermeters for the optimizer and experiment using the config file

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
            # "dataset_name": cfg.DATASET.SOURCE + '2' + cfg.DATASET.TARGET,
            "dataset_name": "rest2" + cfg.DATASET.TARGET,
            # "source": cfg.DATASET.SOURCE,
            "target": cfg.DATASET.TARGET,
            "size_type": cfg.DATASET.SIZE_TYPE,
            "weight_type": cfg.DATASET.WEIGHT_TYPE,
        },
    }
    return config_params


# Based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
def get_model(cfg, dataset, num_channels):
    """
    Builds and returns a model and associated hyperparameters according to the config object passed.

    Args:
        cfg: A YACS config object.
        dataset: A multidomain dataset consisting of source and target datasets.
        num_channels: The number of image channels.
    """

    # setup feature extractor
    feature_network = ResNet50Feature(num_channels)
    # setup classifier
    # feature_dim = feature_network.output_size()
    # target_label = dataset.domain_to_idx[cfg.DATASET.TARGET]
    # classifier_networks = dict()
    # for domain_label_ in dataset.domain_to_idx.values():
    #     if domain_label_ != target_label:
    #         classifier_networks[domain_label_] = ClassNetSmallImage(feature_dim, cfg.DATASET.NUM_CLASSES)

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    # target_label = dataset.domain_to_idx[cfg.DATASET.TARGET]
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

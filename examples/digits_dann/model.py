"""
Define the learning model and configure training parameters.
"""
# Author: Haiping Lu
# Initial Date: 27 July 2020

from copy import deepcopy

import kale.pipeline.domain_adapter as domain_adapter
from kale.embed.image_cnn import SmallCNNFeature
from kale.predict.class_domain_nets import ClassNetSmallImage, DomainNetSmallImage


def get_config(cfg):
    """
    Set the hyperparameters for the optimizer and experiment using the config file

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
            "dataset_name": cfg.DATASET.SOURCE + "2" + cfg.DATASET.TARGET,
            "source": cfg.DATASET.SOURCE,
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
    feature_network = SmallCNNFeature(num_channels)
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = ClassNetSmallImage(feature_dim, cfg.DATASET.NUM_CLASSES)

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    method_params = {}

    method = domain_adapter.Method(cfg.DAN.METHOD)

    if method.is_mmd_method():
        model = domain_adapter.create_mmd_based(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            **method_params,
            **train_params_local,
        )
    else:
        critic_input_size = feature_dim
        # setup critic network
        if method.is_cdan_method():
            if cfg.DAN.USERANDOM:
                critic_input_size = cfg.DAN.RANDOM_DIM
            else:
                critic_input_size = feature_dim * cfg.DATASET.NUM_CLASSES
        critic_network = DomainNetSmallImage(critic_input_size)

        if cfg.DAN.METHOD == "CDAN":
            method_params["use_random"] = cfg.DAN.USERANDOM

        # The following calls kale.loaddata.dataset_access for the first time
        model = domain_adapter.create_dann_like(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            **method_params,
            **train_params_local,
        )
    return model, train_params

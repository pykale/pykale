# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
Define the learning model and configure training parameters.
References from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
"""

from copy import deepcopy

from kale.embed.video_feature_extractor import get_extractor_feat, get_extractor_video
from kale.pipeline import domain_adapter, video_domain_adapter
from kale.predict.class_domain_nets import ClassNetVideo, DomainNetVideo


def get_config(cfg):
    """
    Sets the hyper parameter for the optimizer and experiment using the config file

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
            "optimizer": {"type": cfg.SOLVER.TYPE, "optim_params": {"weight_decay": cfg.SOLVER.WEIGHT_DECAY,},},
        }
    }
    data_params = {
        "data_params": {
            # "dataset_group": cfg.DATASET.NAME,
            "dataset_name": cfg.DATASET.SOURCE + "2" + cfg.DATASET.TARGET,
            "source": cfg.DATASET.SOURCE,
            "target": cfg.DATASET.TARGET,
            "size_type": cfg.DATASET.SIZE_TYPE,
            "weight_type": cfg.DATASET.WEIGHT_TYPE,
            "input_type": cfg.DATASET.INPUT_TYPE,
            "class_type": cfg.DATASET.CLASS_TYPE,
        }
    }
    config_params.update(data_params)
    if config_params["train_params"]["optimizer"]["type"] == "SGD":
        config_params["train_params"]["optimizer"]["optim_params"]["momentum"] = cfg.SOLVER.MOMENTUM
        config_params["train_params"]["optimizer"]["optim_params"]["nesterov"] = cfg.SOLVER.NESTEROV

    return config_params


# Based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/utils/experimentation.py
def get_model(cfg, dataset, dict_num_classes):
    """
    Builds and returns a model and associated hyper parameters according to the config object passed.

    Args:
        cfg: A YACS config object.
        dataset: A multi domain dataset consisting of source and target datasets.
        dict_num_classes (dict): The dictionary of class number for specific dataset.
    """

    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    data_params = config_params["data_params"]
    data_params_local = deepcopy(data_params)
    input_type = data_params_local["input_type"]
    class_type = data_params_local["class_type"]

    # setup feature extractor
    if input_type == "image":
        feature_network, class_feature_dim, domain_feature_dim = get_extractor_video(
            cfg.MODEL.METHOD.upper(), cfg.DATASET.IMAGE_MODALITY, cfg.MODEL.ATTENTION, dict_num_classes["verb"]
        )
    else:
        feature_network, class_feature_dim, domain_feature_dim = get_extractor_feat(
            cfg.DAN.METHOD.upper(),
            cfg.DATASET.IMAGE_MODALITY,
            input_size=1024,
            output_size=256,
            num_segments=cfg.DATASET.NUM_SEGMENTS,
        )

    # setup task classifier
    classifier_network = ClassNetVideo(
        input_size=class_feature_dim, dict_n_class=dict_num_classes, class_type=class_type.lower()
    )

    # setup domain classifier
    method_params = {}
    method = domain_adapter.Method(cfg.DAN.METHOD)

    if method.is_mmd_method():
        model = video_domain_adapter.create_mmd_based_video(
            method=method,
            dataset=dataset,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            class_type=class_type,
            **method_params,
            **train_params_local,
        )
    else:
        critic_input_size = domain_feature_dim
        # setup critic network
        if method.is_cdan_method():
            if cfg.DAN.USERANDOM:
                critic_input_size = cfg.DAN.RANDOM_DIM
            else:
                critic_input_size = domain_feature_dim * dict_num_classes["verb"]
        critic_network = DomainNetVideo(input_size=critic_input_size)

        if cfg.DAN.METHOD == "CDAN":
            method_params["use_random"] = cfg.DAN.USERANDOM

        # The following calls kale.loaddata.dataset_access for the first time
        model = video_domain_adapter.create_dann_like_video(
            method=method,
            dataset=dataset,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            class_type=class_type,
            **method_params,
            **train_params_local,
        )

    return model, train_params

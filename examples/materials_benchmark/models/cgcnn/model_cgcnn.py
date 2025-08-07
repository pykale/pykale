import os
from copy import deepcopy

import torch

from models.cgcnn.CGCNN import CrystalGraphConvNet

from move_to_kale.pipeline.materials_trainer import MaterialsTrainer



def get_config(cfg):
    """
    Sets the hyperparameter for the optimizer and experiment using the config file
    Args:
        cfg: A YACS config object.
    """
    config_params = {
        "train_params": {
            "init_lr": cfg.SOLVER.LR,
            # "adapt_lr": cfg.SOLVER.AD_LR,
            "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            # "lr_gamma": cfg.SOLVER.LR_GAMMA,
            "max_epochs": cfg.SOLVER.EPOCHS,
            "optimizer": {
                "type": cfg.SOLVER.OPTIM,
                "optim_params": {
                    "momentum": cfg.SOLVER.MOMENTUM,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                },
            },
            "layer_freeze": cfg.CGCNN.LAYER_FREEZE,
        },
        # "data_params": {
        #     "num_classes": cfg.DATASET.NUM_CLASSES,
        # },
        "model_params": {
            "orig_atom_fea_len": cfg.GRAPH.ORIG_ATOM_FEA_LEN,
            "nbr_fea_len": cfg.GRAPH.NBR_FEA_LEN,
            "atom_fea_len": cfg.CGCNN.ATOM_FEA_LEN,
            "pos_fea_len": cfg.GRAPH.POS_FEA_LEN,
            "n_conv": cfg.CGCNN.N_CONV,
            "h_fea_len": cfg.CGCNN.H_FEA_LEN,
            "n_h": cfg.CGCNN.N_H,
            "classification": cfg.SOLVER.TASK == "classification",
            "feature_fusion": cfg.CGCNN.FEATURE_FUSION,
            
        },
    }

    return config_params

def get_cgcnn_model(cfg):
    """
    Builds and returns a model according to the config object passed.

    Args:
        cfg: A YACS config object.
    """
    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    model_params = config_params["model_params"]
    model_params_local = deepcopy(model_params)

    model = CrystalGraphConvNet(
        orig_atom_fea_len=model_params_local["orig_atom_fea_len"],
        nbr_fea_len=model_params_local["nbr_fea_len"],
        atom_fea_len=model_params_local["atom_fea_len"],
        n_conv=model_params_local["n_conv"],
        h_fea_len=model_params_local["h_fea_len"],
        n_h=model_params_local["n_h"],
        classification=model_params_local["classification"]
    )

    trainer = MaterialsTrainer(model=model, **train_params_local)

    # Optionally resume from a checkpoint
    if hasattr(cfg, 'PRETRAINED_MODEL_PATH') and os.path.isfile(cfg.PRETRAINED_MODEL_PATH):
        print("=> loading checkpoint '{}'".format(cfg.PRETRAINED_MODEL_PATH))
        checkpoint = torch.load(cfg.PRETRAINED_MODEL_PATH)
        trainer.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            trainer.configure_optimizers()[0].load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(cfg.PRETRAINED_MODEL_PATH,
                                                            checkpoint.get('epoch', 'unknown')))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.MODEL.PRETRAINED_MODEL_PATH))

    return trainer
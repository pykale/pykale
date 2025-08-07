import os
from copy import deepcopy

import torch

from models.cartnet.CartNet import CartNet
from move_to_kale.pipeline.materials_trainer import MaterialsTrainer


def get_config(cfg):
    """
    Sets the hyperparameters for the optimizer and experiment using the config file
    Args:
        cfg: A YACS config object.
    """
    config_params = {
        "train_params": {
            "init_lr": cfg.SOLVER.LR,
            "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            "max_epochs": cfg.SOLVER.EPOCHS,
            "optimizer": {
                "type": cfg.SOLVER.OPTIM,
                "optim_params": {
                    "momentum": cfg.SOLVER.MOMENTUM,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                },
            },
            # "layer_freeze": cfg.CARTNET.LAYER_FREEZE,
        },
        "model_params": {
            "dim_in": cfg.CARTNET.DIM_IN,
            "dim_rbf": cfg.CARTNET.DIM_RBF,
            "num_layers": cfg.CARTNET.NUM_LAYERS,
            "invariant": cfg.CARTNET.INVARIANT,
            "temperature": cfg.CARTNET.TEMPERATURE,
            "use_envelope": cfg.CARTNET.USE_ENVELOPE,
            "atom_types": cfg.CARTNET.ATOM_TYPES,
            "radius": cfg.MODEL.RADIUS,
        },
    }

    return config_params

def get_cartnet_model(cfg):
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

    model = CartNet(dim_in=model_params_local["dim_in"],
                        dim_rbf=model_params_local["dim_rbf"], 
                        num_layers=model_params_local["num_layers"], 
                        radius=model_params_local["radius"],
                        invariant=model_params_local["invariant"], 
                        temperature=model_params_local["temperature"], 
                        use_envelope=model_params_local["use_envelope"],
                        atom_types=model_params_local["atom_types"],
                        cholesky=False
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
from copy import deepcopy

from move_to_kale.pipeline.materials_trainer import MaterialsTrainer
from models.leftnet.leftnet_prop import LEFTNetProp
from models.leftnet.leftnet_z import LEFTNetZ



def get_config(cfg):
    """
    Sets the hyperparameters for the optimizer and experiment using the config file
    Args:
        cfg: A YACS config object.
    """
    config_params = {
        "leftnet_params": {
            "cutoff": cfg.LEFTNET.CUTOFF,
            "hidden_channels": cfg.LEFTNET.HIDDEN_CHANNELS,
            "num_layers": cfg.LEFTNET.NUM_LAYERS,
            "num_radial": cfg.LEFTNET.NUM_RADIAL,
            "regress_forces": cfg.LEFTNET.REGRESS_FORCES,
            "use_pbc": cfg.LEFTNET.USE_PBC,
            "otf_graph": cfg.LEFTNET.OTF_GRAPH,
            "output_dim": cfg.LEFTNET.OUTPUT_DIM,
            # "atom_fea_dim": cfg.LEFTNET.ATOM_FEA_DIM,
  
        },
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
            "layer_freeze": cfg.LEFTNET.LAYER_FREEZE,
        },
         "encoding": cfg.LEFTNET.ENCODING,
        
    }

    return config_params


def get_leftnet_model(cfg):
    config_params = get_config(cfg)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    leftnet_params = config_params["leftnet_params"]
    leftnet_params_local = deepcopy(leftnet_params)
    encoding = config_params["encoding"]

    num_atoms = 1
    bond_feat_dim = leftnet_params_local.get("num_gaussians", 50)

    if encoding == "z":
        model = LEFTNetZ(
            atom_fea_dim=95, num_targets=leftnet_params_local.get("output_dim"), **leftnet_params_local  # not used  # not used
        )
    elif encoding == "prop":
        model = LEFTNetProp(
            bond_feat_dim=bond_feat_dim, num_targets=leftnet_params_local.get("output_dim"), **leftnet_params_local  # not used  # not used
        )
    # return model
    trainer = MaterialsTrainer(model=model, **train_params_local)

    return trainer

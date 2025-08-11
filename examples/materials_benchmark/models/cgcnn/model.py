import os
from copy import deepcopy

import torch

from models.cgcnn.CGCNN import CrystalGraphConvNet
from ..leftnet.leftnet_z import LEFTNetZ
from ..leftnet.leftnet_prop import LEFTNetProp
from ..cartnet.CartNet import CartNet

from move_to_kale.pipeline.materials_trainer import MaterialsTrainer



def get_config(cfg):
    """
    Sets the hyperparameter for the optimizer and experiment using the config file
    Args:
        cfg: A YACS config object.
    """
    # -------------------- train params (same for all models) --------------------
    config_params = {
        "train_params": {
            "init_lr": cfg.SOLVER.LR,
            "max_epochs": cfg.SOLVER.EPOCHS,
            "optimizer": {
                "type": cfg.SOLVER.OPTIM,  # 'SGD' | 'Adam'
                "optim_params": {
                    "momentum": cfg.SOLVER.MOMENTUM,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                },
            },
            # If you later need LR milestones/gamma, just uncomment these:
            # "lr_milestones": cfg.SOLVER.LR_MILESTONES,
            # "lr_gamma": cfg.SOLVER.LR_GAMMA,
            # "layer_freeze": cfg.CGCNN.LAYER_FREEZE,  # example: move in when needed
        },
        "model_params": {},  # filled below based on cfg.MODEL.NAME
    }

    name = str(cfg.MODEL.NAME).lower()

    # -------------------- model-specific params --------------------
    if name == "cgcnn":
        config_params["model_params"] = {
            "orig_atom_fea_len": cfg.GRAPH.ORIG_ATOM_FEA_LEN,
            "nbr_fea_len": cfg.GRAPH.NBR_FEA_LEN,
            "atom_fea_len": cfg.CGCNN.ATOM_FEA_LEN,
            "pos_fea_len": cfg.GRAPH.POS_FEA_LEN,
            "n_conv": cfg.CGCNN.N_CONV,
            "h_fea_len": cfg.CGCNN.H_FEA_LEN,
            "n_h": cfg.CGCNN.N_H,
            "classification": (cfg.SOLVER.TASK == "classification"),
            "feature_fusion": cfg.CGCNN.FEATURE_FUSION,
            # "layer_freeze": cfg.CGCNN.LAYER_FREEZE,  # add back if your builder uses it
        }

    elif name == "leftnet":
        config_params["model_params"] = {
            "orig_atom_fea_len": cfg.GRAPH.ORIG_ATOM_FEA_LEN,
            "nbr_fea_len": cfg.GRAPH.NBR_FEA_LEN,
            "pos_fea_len": cfg.GRAPH.POS_FEA_LEN,
            "cutoff": cfg.LEFTNET.CUTOFF,
            "hidden_channels": cfg.LEFTNET.HIDDEN_CHANNELS,
            "num_layers": cfg.LEFTNET.NUM_LAYERS,
            "num_radial": cfg.LEFTNET.NUM_RADIAL,
            "regress_forces": cfg.LEFTNET.REGRESS_FORCES,
            "use_pbc": cfg.LEFTNET.USE_PBC,
            "otf_graph": cfg.LEFTNET.OTF_GRAPH,
            "output_dim": cfg.LEFTNET.OUTPUT_DIM,
            "encoding": cfg.LEFTNET.ENCODING,  # ['one-hot', 'none']
            # "classification": (cfg.SOLVER.TASK == "classification"),
            # "layer_freeze": cfg.LEFTNET.LAYER_FREEZE,
        }

    elif name == "cartnet":
        config_params["model_params"] = {
            "orig_atom_fea_len": cfg.GRAPH.ORIG_ATOM_FEA_LEN,
            "nbr_fea_len": cfg.GRAPH.NBR_FEA_LEN,
            "pos_fea_len": cfg.GRAPH.POS_FEA_LEN,
            "dim_in": cfg.CARTNET.DIM_IN,
            "dim_rbf": cfg.CARTNET.DIM_RBF,
            "num_layers": cfg.CARTNET.NUM_LAYERS,
            "invariant": cfg.CARTNET.INVARIANT,
            "temperature": cfg.CARTNET.TEMPERATURE,
            "use_envelope": cfg.CARTNET.USE_ENVELOPE,
            "atom_types": cfg.CARTNET.ATOM_TYPES,
            # "classification": (cfg.SOLVER.TASK == "classification"),
        }

    # classic ML baselines 
    elif name in {"random_forest", "svm", "linear_regression"}:
        config_params["model_params"] = {
        
            "orig_atom_fea_len": cfg.GRAPH.ORIG_ATOM_FEA_LEN,
            "nbr_fea_len": cfg.GRAPH.NBR_FEA_LEN,
            "pos_fea_len": cfg.GRAPH.POS_FEA_LEN,
        }

    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

    return config_params

def build_cgcnn(cfg, train_params, model_params):
    model = CrystalGraphConvNet(
        orig_atom_fea_len=model_params["orig_atom_fea_len"],
        nbr_fea_len=model_params["nbr_fea_len"],
        atom_fea_len=model_params["atom_fea_len"],
        n_conv=model_params["n_conv"],
        h_fea_len=model_params["h_fea_len"],
        n_h=model_params["n_h"],
        classification=model_params["classification"]
    )
    return MaterialsTrainer(feature_extractor=model, **train_params)

def build_leftnet(cfg, train_params, model_params):
    encoding = model_params.get("encoding", "none")
    bond_feat_dim = model_params.get("num_radial", 50)

    if encoding == "z":
        model = LEFTNetZ(
            atom_fea_dim=95,
            num_targets=model_params.get("output_dim", 1),
            **model_params
        )
    elif encoding == "prop":
        model = LEFTNetProp(
            bond_feat_dim=bond_feat_dim,
            num_targets=model_params.get("output_dim", 1),
            **model_params
        )
    else:
        raise ValueError(f"Unsupported LEFTNet encoding: {encoding}")

    return MaterialsTrainer(feature_extractor=model, **train_params)

def build_cartnet(cfg, train_params, model_params):

    model = CartNet(dim_in=model_params["dim_in"],
                    dim_rbf=model_params["dim_rbf"], 
                    num_layers=model_params["num_layers"], 
                    radius=model_params["radius"],
                    invariant=model_params["invariant"], 
                    temperature=model_params["temperature"], 
                    use_envelope=model_params["use_envelope"],
                    atom_types=model_params["atom_types"],
                    cholesky=False
                )
    return MaterialsTrainer(feature_extractor=model, **train_params)

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

    model_name = cfg.MODEL.NAME
    if model_name == "cgcnn":
        trainer = build_cgcnn(cfg, train_params_local, model_params_local)
    # elif model_name == "graphormer":
    #     trainer = build_graphormer(cfg, train_params, model_params)
    # elif model_name == "megnet":
    #     trainer = build_megnet(cfg, train_params, model_params)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")
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



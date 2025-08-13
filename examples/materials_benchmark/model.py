import os
from copy import deepcopy

import torch

from kale.embed.gcn import CrystalGraphConvNet
from kale.embed.gcn import LEFTNetZ, LEFTNetProp

from kale.embed.gcn import CartNet

from kale.pipeline.base_nn_trainer import RegressionTrainer



def get_config(cfg, atom_fea_len, nbr_fea_len, pos_fea_len):
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
        },
        "model_params": {},  
    }

    name = str(cfg.MODEL.NAME).lower()

    # -------------------- model-specific params --------------------
    if name == "cgcnn":
        config_params["model_params"] = {
            "orig_atom_fea_len": atom_fea_len,
            "nbr_fea_len": nbr_fea_len,
            "atom_fea_len": cfg.CGCNN.ATOM_FEA_LEN,
            "pos_fea_len": pos_fea_len,
            "n_conv": cfg.CGCNN.N_CONV,
            "h_fea_len": cfg.CGCNN.H_FEA_LEN,
            "n_h": cfg.CGCNN.N_H,
            "feature_fusion": cfg.CGCNN.FEATURE_FUSION,
            # "layer_freeze": cfg.CGCNN.LAYER_FREEZE,  # add back if your builder uses it
        }

    elif name == "leftnet":
        config_params["model_params"] = {
            "atom_fea_dim": atom_fea_len,
            # "nbr_fea_len": nbr_fea_len,
            # "pos_fea_len": pos_fea_len,
            "cutoff": cfg.LEFTNET.CUTOFF,
            "hidden_channels": cfg.LEFTNET.HIDDEN_CHANNELS,
            "num_layers": cfg.LEFTNET.NUM_LAYERS,
            "num_radial": cfg.LEFTNET.NUM_RADIAL,
            "regress_forces": cfg.LEFTNET.REGRESS_FORCES,
            "use_pbc": cfg.LEFTNET.USE_PBC,
            "otf_graph": cfg.LEFTNET.OTF_GRAPH,
            "output_dim": cfg.LEFTNET.OUTPUT_DIM,
            "encoding": cfg.LEFTNET.ENCODING,  # ['one-hot', 'none']
        
        }

    elif name == "cartnet":
        config_params["model_params"] = {
            
            "dim_in": cfg.CARTNET.DIM_IN,
            "dim_rbf": cfg.CARTNET.DIM_RBF,
            "num_layers": cfg.CARTNET.NUM_LAYERS,
            "invariant": cfg.CARTNET.INVARIANT,
            "temperature": cfg.CARTNET.TEMPERATURE,
            "use_envelope": cfg.CARTNET.USE_ENVELOPE,
            "atom_types": cfg.CARTNET.ATOM_TYPES,
       
        }

    # classic ML baselines 
    elif name in {"random_forest", "svm", "linear_regression"}:
        config_params["model_params"] = {
        
            "orig_atom_fea_len": atom_fea_len,
            "nbr_fea_len": nbr_fea_len,
            "pos_fea_len": pos_fea_len,
        }

    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

    return config_params

def build_cgcnn(train_params, model_params):
    model = CrystalGraphConvNet(
        orig_atom_fea_len=model_params["orig_atom_fea_len"],
        nbr_fea_len=model_params["nbr_fea_len"],
        atom_fea_len=model_params["atom_fea_len"],
        n_conv=model_params["n_conv"],
        h_fea_len=model_params["h_fea_len"],
        n_h=model_params["n_h"]
    )
    return RegressionTrainer(feature_extractor=model, **train_params)

def build_leftnet(train_params, model_params):
    encoding = model_params.get("encoding", "none")
    # bond_feat_dim = model_params.get("num_radial", 50)
    # remove encoding from model_params
    model_params = {k: v for k, v in model_params.items() if k != "encoding"}
    if encoding == "z":
        model = LEFTNetZ(
            atom_fea_dim=95,
            num_targets=model_params.get("output_dim", 1),
            **model_params
        )
    elif encoding == "prop":
        model = LEFTNetProp(
            # atom_fea_dim=model_params["atom_fea_dim"],
            num_targets=model_params.get("output_dim", 1),
            **model_params
        )
    else:
        raise ValueError(f"Unsupported LEFTNet encoding: {encoding}")

    return RegressionTrainer(feature_extractor=model, **train_params)

def build_cartnet(train_params, model_params):

    model = CartNet(**model_params)
    return RegressionTrainer(feature_extractor=model, **train_params)

def get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len):
    """
    Builds and returns a model according to the config object passed.

    Args:
        cfg: A YACS config object.
    """
    config_params = get_config(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)
    train_params = config_params["train_params"]
    train_params_local = deepcopy(train_params)
    model_params = config_params["model_params"]
    model_params_local = deepcopy(model_params)

    model_name = cfg.MODEL.NAME
    if model_name == "cgcnn":
        trainer = build_cgcnn(train_params_local, model_params_local)
    elif model_name == "leftnet":
        trainer = build_leftnet(train_params_local, model_params_local)
    elif model_name == "cartnet":
        trainer = build_cartnet(train_params_local, model_params_local)
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



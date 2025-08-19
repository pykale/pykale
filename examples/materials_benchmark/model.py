import os
from copy import deepcopy

import torch

from kale.embed.gcn import CartNet, CrystalGCN, LEFTNetProp, LEFTNetZ
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
                "type": cfg.SOLVER.OPTIM,
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
            "n_conv": cfg.CGCNN.N_CONV,
            "h_fea_len": cfg.CGCNN.H_FEA_LEN,
            "n_h": cfg.CGCNN.N_H,
        }

    elif name == "leftnet":
        encoding = cfg.LEFTNET.ENCODING
        atom_fea_dim = atom_fea_len if encoding == "prop" else 95  # if leftnet-prop, atom_fea_len=len of one-hot vector

        config_params["model_params"] = {
            "atom_fea_dim": atom_fea_dim,
            "num_targets": cfg.LEFTNET.TARGET_DIM,
            "cutoff": cfg.LEFTNET.CUTOFF,
            "hidden_channels": cfg.LEFTNET.HIDDEN_CHANNELS,
            "num_layers": cfg.LEFTNET.NUM_LAYERS,
            "num_radial": cfg.LEFTNET.NUM_RADIAL,
            "regress_forces": cfg.LEFTNET.REGRESS_FORCES,
            "use_pbc": cfg.LEFTNET.USE_PBC,
            "otf_graph": cfg.LEFTNET.OTF_GRAPH,
            "output_dim": cfg.LEFTNET.OUTPUT_DIM,
            "encoding": encoding,
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


def build_cgcnn(model_params):
    model = CrystalGCN(**model_params)
    return model


def build_leftnet(model_params):
    encoding = model_params.get("encoding", "none")
    # remove encoding key
    model_params = {k: v for k, v in model_params.items() if k != "encoding"}

    if encoding == "z":
        model = LEFTNetZ(**model_params)
    elif encoding == "prop":
        model = LEFTNetProp(**model_params)
    else:
        raise ValueError(f"Unsupported LEFTNet encoding: {encoding}")

    return model


def build_cartnet(model_params):
    model = CartNet(**model_params)
    return model


def get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len):
    """
    Build trainer by cfg and optionally load from checkpoint.
    """
    config_params = get_config(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)
    train_params = deepcopy(config_params["train_params"])
    model_params = deepcopy(config_params["model_params"])

    # build bare model
    model_name = cfg.MODEL.NAME
    if model_name == "cgcnn":
        model = build_cgcnn(model_params)
    elif model_name == "leftnet":
        model = build_leftnet(model_params)
    elif model_name == "cartnet":
        model = build_cartnet(model_params)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

    # wrap into trainer
    trainer = RegressionTrainer(feature_extractor=model, **train_params)

    # load checkpoint if exists
    pretrained_path = getattr(cfg, "PRETRAINED_MODEL_PATH", None) or getattr(cfg.MODEL, "PRETRAINED_MODEL_PATH", None)

    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"=> loading checkpoint '{pretrained_path}'")
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        missing, unexpected = trainer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading state_dict: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading state_dict: {unexpected}")

        print(
            f"=> loaded checkpoint '{pretrained_path}' "
            f"(epoch {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'})"
        )
    else:
        print(f"=> no checkpoint found at '{pretrained_path}'")

    return trainer

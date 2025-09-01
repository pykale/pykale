import os
from copy import deepcopy

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from kale.embed.model_lib.cartnet import CartNet
from kale.embed.model_lib.cgcnn import CrystalGCN
from kale.embed.model_lib.leftnet import LEFTNetProp, LEFTNetZ
from kale.evaluate.metrics import mean_relative_error
from kale.loaddata.materials_datasets import CIFData
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
            # "regress_forces": cfg.LEFTNET.REGRESS_FORCES,
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


# TODO: move to kale.pipeline.traditional_ml
def run_traditional_ml(model_name, x_train, y_train, x_valid, y_valid, seed=42, x_test=None, y_test=None):
    if model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    elif model_name == "linear_regression":
        model = LinearRegression()
    else:
        model = SVR()

    model.fit(x_train, y_train)
    val_pred = model.predict(x_valid)

    y_valid, y_test, val_pred = [torch.as_tensor(a, dtype=torch.float32) for a in (y_valid, y_test, val_pred)]

    print(
        "Validation - MAE: {:.4f}, MRE: {:.4f}, R²: {:.4f}".format(
            mean_absolute_error(y_valid, val_pred),
            mean_relative_error(y_valid, val_pred),
            r2_score(y_valid, val_pred),
        )
    )

    test_pred = torch.as_tensor(model.predict(x_test), dtype=torch.float32)
    print(
        "Test - MAE: {:.4f}, MRE: {:.4f}, R²: {:.4f}".format(
            mean_absolute_error(y_test, test_pred),
            mean_relative_error(y_test, test_pred),
            r2_score(y_test, test_pred),
        )
    )


# TODO: move to kale.prepdata.tabular_transform
def split_data_by_kfold(data, n_splits, seed=None, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    train_data_list = []  # list of DataFrame
    val_data_list = []
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index].reset_index(drop=True)
        val_data = data.iloc[val_index].reset_index(drop=True)
        train_data_list.append(train_data)
        val_data_list.append(val_data)
    return train_data_list, val_data_list

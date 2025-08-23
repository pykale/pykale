import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from kale.loaddata.materials_datasets import CIFData
from kale.prepdata.materials_features import extract_features


def arg_parse():
    parser = argparse.ArgumentParser(description="Bandgap Benchmark (single split)")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help=(
            "gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU. "
            "str(x) for the first x GPUs. str(-1)/int(-1) for all GPUs"
        ),
    )
    return parser.parse_args()


def main():
    args = arg_parse()

    # --- config ---
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # --- seed ----
    import random

    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SOLVER.SEED)

    target_key = "bg"

    # --- datasets ---
    train_dataset = CIFData(
        target_path=cfg.DATASET.TRAIN,
        cif_folder=cfg.MODEL.CIF_FOLDER,
        init_file=cfg.MODEL.INIT_FILE,
        max_nbrs=cfg.MODEL.MAX_NBRS,
        radius=cfg.MODEL.RADIUS,
        randomize=cfg.SOLVER.RANDOMIZE,
        target_key=target_key,
    )

    val_dataset = CIFData(
        target_path=cfg.DATASET.VAL,
        cif_folder=cfg.MODEL.CIF_FOLDER,
        init_file=cfg.MODEL.INIT_FILE,
        max_nbrs=cfg.MODEL.MAX_NBRS,
        radius=cfg.MODEL.RADIUS,
        randomize=False,
        target_key=target_key,
    )

    test_dataset = CIFData(
        target_path=cfg.DATASET.TEST,
        cif_folder=cfg.MODEL.CIF_FOLDER,
        init_file=cfg.MODEL.INIT_FILE,
        max_nbrs=cfg.MODEL.MAX_NBRS,
        radius=cfg.MODEL.RADIUS,
        randomize=False,
        target_key=target_key,
    )

    # --- loaders ---
    train_loader = DataLoader(
        train_dataset,
        collate_fn=CIFData.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.SOLVER.WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=CIFData.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=CIFData.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.WORKERS,
    )

    # --- feature dimensions ---
    sample = train_dataset[0]
    atom_fea_len, nbr_fea_len, pos_fea_len = (
        sample.atom_fea.shape[-1],
        sample.nbr_fea.shape[-1],
        sample.positions.shape[-1],
    )

    # --- traditional ML branch ---
    if cfg.MODEL.NAME in ["random_forest", "linear_regression", "svm"]:
        print(f"Training classic model: {cfg.MODEL.NAME}")
        x_train, y_train = extract_features(train_dataset)
        x_valid, y_valid = extract_features(val_dataset)

        if cfg.MODEL.NAME == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=cfg.SOLVER.SEED)
        elif cfg.MODEL.NAME == "linear_regression":
            model = LinearRegression()
        else:
            model = SVR()

        model.fit(x_train, y_train)
        val_pred = model.predict(x_valid)
        print(
            "Validation - MAE: {:.4f}, MSE: {:.4f}, R²: {:.4f}".format(
                mean_absolute_error(y_valid, val_pred),
                mean_squared_error(y_valid, val_pred),
                r2_score(y_valid, val_pred),
            )
        )
        if test_loader is not None:
            X_test, y_test = extract_features(test_dataset)
            test_pred = model.predict(X_test)
            print(
                "Test       - MAE: {:.4f}, MSE: {:.4f}, R²: {:.4f}".format(
                    mean_absolute_error(y_test, test_pred),
                    mean_squared_error(y_test, test_pred),
                    r2_score(y_test, test_pred),
                )
            )
        return

    # --- GNNs ---
    model = get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)

    # load pretrained if specified
    pretrained_path = getattr(cfg.MODEL, "PRETRAINED_MODEL_PATH", "")
    if pretrained_path:
        if os.path.isfile(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded: {pretrained_path}")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)
        else:
            print(f"PRETRAINED_MODEL_PATH not found: {pretrained_path} (skip)")

    # logger & callbacks (inline)
    log_dir_name = cfg.LOGGING.LOG_DIR_NAME or f"experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"{cfg.LOGGING.LOG_DIR}/{log_dir_name}"
    os.makedirs(log_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=cfg.LOGGING.PROJECT_NAME,
        name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    try:
        wandb_logger.experiment.config.update(cfg)
    except Exception:
        pass

    best_mre_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_mre",
        mode="min",
        save_top_k=1,
        filename="best-mre-{epoch:02d}-{val_mre:.4f}",
    )
    last_ckpt = ModelCheckpoint(dirpath=log_dir, save_last=True, filename="last-{epoch:02d}")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.EPOCHS,
        accelerator="gpu" if int(args.devices) > 0 and torch.cuda.is_available() else "cpu",
        devices=args.devices if int(args.devices) > 0 else None,
        logger=wandb_logger,
        callbacks=[best_mre_ckpt, last_ckpt, lr_monitor],
    )

    # train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test best checkpoint
    trainer.test(ckpt_path="best", dataloaders=val_loader)
    if test_loader is not None:
        trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    main()

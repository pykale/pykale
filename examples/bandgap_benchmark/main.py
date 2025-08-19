import argparse
import json
import os
import random
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from kale.evaluate.metrics import mean_relative_error
from kale.loaddata.materials_datasets import CIFData
from kale.prepdata.materials_features import extract_features


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="CGCNN")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--pretrain", action="store_true", help="Enable pretraining mode using the full dataset")
    args = parser.parse_args()
    return args


def load_data(cfg):
    with open(cfg.DATASET.TRAIN, "r") as f:
        train_data = json.load(f)
        train_data = pd.DataFrame.from_dict(train_data, orient="index")
        train_data.reset_index(inplace=True)
        train_data.rename(columns={"index": "mpids"}, inplace=True)
    with open(cfg.DATASET.VAL, "r") as f:
        val_data = json.load(f)
        val_data = pd.DataFrame.from_dict(val_data, orient="index")
        val_data.reset_index(inplace=True)
        val_data.rename(columns={"index": "mpids"}, inplace=True)
    return train_data, val_data


def prepare_datasets(cfg, train_fold, val_fold):
    train_fold = shuffle(train_fold[["mpids", "bg"]], random_state=cfg.SOLVER.SEED)
    train_dataset = CIFData(
        train_fold,
        cfg.MODEL.CIF_FOLDER,
        cfg.MODEL.INIT_FILE,
        cfg.MODEL.MAX_NBRS,
        cfg.MODEL.RADIUS,
        cfg.SOLVER.RANDOMIZE,
    )

    val_dataset = CIFData(
        val_fold[["mpids", "bg"]],
        cfg.MODEL.CIF_FOLDER,
        cfg.MODEL.INIT_FILE,
        cfg.MODEL.MAX_NBRS,
        cfg.MODEL.RADIUS,
        cfg.SOLVER.RANDOMIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.WORKERS,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_model(model, path):
    if not os.path.isfile(path):
        print(f"=> no checkpoint found at '{path}'")
        return model

    print(f"=> loading checkpoint '{path}'")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.") and not k.startswith("model.feature_extractor."):
            new_k = k.replace("model.", "model.feature_extractor.")
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading state_dict: {unexpected}")

    if "optimizer" in checkpoint:
        try:
            model.configure_optimizers()[0].load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    print(f"=> loaded checkpoint '{path}' (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def setup_logger(cfg, fold):
    log_dir_name = cfg.LOGGING.LOG_DIR_NAME
    if log_dir_name is None:
        log_dir_name = f"experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{fold}"
    log_dir = f"{cfg.LOGGING.LOG_DIR}/{log_dir_name}"

    wandb_logger = WandbLogger(
        project="bandgap-project",
        name="experiment_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")),
    )

    wandb_logger.experiment.config.update(cfg)

    return wandb_logger, log_dir


def setup_trainer(cfg, args, wandb_logger, log_dir):
    # Save best model based on val_mre
    best_mre_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_mre",
        mode="min",
        save_top_k=1,
        filename="best-mre-{epoch:02d}-{val_mre:.4f}",
    )

    # Save best model based on val_mae
    best_mae_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        filename="best-mae-{epoch:02d}-{val_mae:.4f}",
    )

    # Always save the last model
    last_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        save_last=True,
        filename="last-{epoch:02d}",
    )

    # Save a checkpoint every 20 epochs
    periodic_checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        every_n_epochs=20,
        save_top_k=-1,
        filename="epoch-{epoch:02d}",
    )

    # metrics_callback = MetricsCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.EPOCHS,
        accelerator="gpu" if int(args.devices) > 0 else "cpu",
        devices=args.devices if int(args.devices) > 0 else None,
        logger=wandb_logger,
        callbacks=[
            best_mre_checkpoint,
            best_mae_checkpoint,
            last_checkpoint,
            periodic_checkpoint,
            lr_monitor,
            # metrics_callback,
        ],
    )
    return trainer, best_mre_checkpoint


def evaluate_model(trainer, model, val_loader, fold):
    val_results = trainer.test(model, dataloaders=val_loader)
    print(f"Validation Results (Fold {str(fold)}): {val_results}")
    return val_results


def test_model(cfg, trainer, model, fold):
    if cfg.DATASET.VAL:
        with open(cfg.DATASET.VAL, "r") as f:
            test_data = json.load(f)
            test_data = pd.DataFrame.from_dict(test_data, orient="index")
            test_data.reset_index(inplace=True)
            test_data.rename(columns={"index": "mpids"}, inplace=True)
        test_dataset = CIFData(
            test_data[["mpids", "bg"]],
            cfg.MODEL.CIF_FOLDER,
            cfg.MODEL.INIT_FILE,
            cfg.MODEL.MAX_NBRS,
            cfg.MODEL.RADIUS,
            cfg.SOLVER.RANDOMIZE,
        )
        test_loader = DataLoader(
            test_dataset,
            collate_fn=test_dataset.collate_fn,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.SOLVER.WORKERS,
        )
        test_results = trainer.test(model, dataloaders=test_loader)
        print(f"Test Results (Fold {str(fold)}): {test_results}")
        return test_results


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Set random seed for reproducibility
    set_random_seed(cfg.SOLVER.SEED)

    train_data, val_data = load_data(cfg)

    if args.pretrain:
        print("Pretraining mode enabled. Splitting training data into k folds for validation.")

        # Split and prepare data
        kf = KFold(n_splits=cfg.SOLVER.NUM_FOLDS, shuffle=True, random_state=cfg.SOLVER.SEED)
        train_idx, val_idx = next(kf.split(train_data))
        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]
        train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(cfg, train_fold, val_fold)

        # Extract feature dimensions
        sample = train_dataset[0]
        atom_fea_len, nbr_fea_len, pos_fea_len = (
            sample.atom_fea.shape[-1],
            sample.nbr_fea.shape[-1],
            sample.positions.shape[-1],
        )

        # Setup model and trainer
        model = get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)
        wandb_logger, log_dir = setup_logger(cfg, "pretrain")
        trainer, checkpoint_callback = setup_trainer(cfg, args, wandb_logger, log_dir)

        # Resume from checkpoint if exists
        ckpt_path = cfg.MODEL.PRETRAINED_MODEL_PATH if os.path.exists(cfg.MODEL.PRETRAINED_MODEL_PATH) else None

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

        # Evaluate best model
        best_model_path = checkpoint_callback.best_model_path
        model.load_state_dict(torch.load(best_model_path)["state_dict"])
        evaluate_model(trainer, model, val_loader, "pretrain")
        test_model(cfg, trainer, model, "pretrain")

    else:
        num_folds = cfg.SOLVER.NUM_FOLDS  # Add this to your configuration (e.g., 5 or 10)

        # Initialize K-Fold Cross-Validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=cfg.SOLVER.SEED)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            print(f"\nFold {fold + 1}/{num_folds}")
            train_fold = train_data.iloc[train_idx]
            val_fold = train_data.iloc[val_idx]

            # Prepare datasets and loaders
            train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(cfg, train_fold, val_fold)

            sample = train_dataset[0]
            atom_fea_len, nbr_fea_len, pos_fea_len = (
                sample.atom_fea.shape[-1],
                sample.nbr_fea.shape[-1],
                sample.positions.shape[-1],
            )

            if cfg.MODEL.NAME in ["random_forest", "linear_regression", "svm"]:
                print(f"Training {cfg.MODEL.NAME} model...")
                X_train, y_train = extract_features(train_dataset)
                X_val, y_val = extract_features(val_dataset)

                if cfg.MODEL.NAME == "random_forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=cfg.SOLVER.SEED)
                elif cfg.MODEL.NAME == "linear_regression":
                    model = LinearRegression()
                elif cfg.MODEL.NAME == "svm":
                    model = SVR()

                model.fit(X_train, y_train)

                relative_error_scorer = make_scorer(
                    lambda yt, yp: float(
                        mean_relative_error(
                            torch.as_tensor(yt, dtype=torch.float32), torch.as_tensor(yp, dtype=torch.float32)
                        )
                    ),
                    greater_is_better=False,
                )
                result = permutation_importance(
                    estimator=model,
                    X=X_val,
                    y=y_val,
                    scoring=relative_error_scorer,
                    n_repeats=10,
                    random_state=cfg.SOLVER.SEED,
                )

                importances_mean = result.importances_mean
                feature_importances_file = os.path.join(cfg.LOGGING.LOG_DIR, f"feature_importances_fold_{fold + 1}.txt")

                # Ensure directory exists
                os.makedirs(cfg.LOGGING.LOG_DIR, exist_ok=True)

                with open(feature_importances_file, "w") as f:
                    f.write("Feature Importances:\n")
                    for i, imp in enumerate(importances_mean):
                        f.write(f"Feature {i:2d}: {imp}\n")

                val_predictions = model.predict(X_val)
                mae = mean_absolute_error(y_val, val_predictions)
                mse = mean_squared_error(y_val, val_predictions)
                mre = np.mean(np.abs((y_val - val_predictions) / y_val))
                r2 = r2_score(y_val, val_predictions)
                print(
                    f"Fold {fold + 1} {cfg.MODEL.NAME} Validation Result- MAE: {mae:}, MSE: {mse:}, MRE: {mre:}, R²: {r2:}"
                )

                if cfg.DATASET.VAL:
                    with open(cfg.DATASET.VAL, "r") as f:
                        test_data = json.load(f)
                        test_data = pd.DataFrame.from_dict(test_data, orient="index").reset_index()
                        test_data.rename(columns={"index": "mpids"}, inplace=True)
                    test_dataset = CIFData(
                        test_data[["mpids", "bg"]],
                        cfg.MODEL.CIF_FOLDER,
                        cfg.MODEL.INIT_FILE,
                        cfg.MODEL.MAX_NBRS,
                        cfg.MODEL.RADIUS,
                        cfg.SOLVER.RANDOMIZE,
                    )
                    X_test, y_test = extract_features(test_dataset)
                    test_predictions = model.predict(X_test)
                    test_mae = mean_absolute_error(y_test, test_predictions)
                    test_mse = mean_squared_error(y_test, test_predictions)
                    test_mre = np.mean(np.abs((y_test - test_predictions) / y_test))
                    test_r2 = r2_score(y_test, test_predictions)
                    print(
                        f"Fold {fold + 1} {cfg.MODEL.NAME} Test Result- MAE: {test_mae:}, MSE: {test_mse:}, MRE: {test_mre:}, R²: {test_r2:}"
                    )
                continue

            model = get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)

            wandb_logger, log_dir = setup_logger(cfg, fold)
            trainer, checkpoint_callback = setup_trainer(cfg, args, wandb_logger, log_dir)

            # Train the model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Load the best model for evaluation
            best_model_path = checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(best_model_path)["state_dict"])

            # Evaluate the model
            evaluate_model(trainer, model, val_loader, fold)

            # Test on evaluation dataset if applicable
            test_model(cfg, trainer, model, fold)


if __name__ == "__main__":
    main()

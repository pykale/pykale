import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model, run_traditional_ml, split_data_by_kfold
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from kale.loaddata.materials_datasets import CrystalDataset
from kale.prepdata.materials_features import extract_features
from kale.utils.format_transformation import load_json_to_df
from kale.utils.seed import set_seed


def arg_parse():
    parser = argparse.ArgumentParser(description="Bandgap Benchmark (single split)")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--pretrain", action="store_true", help="enable pretraining")
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
    set_seed(cfg.SOLVER.SEED)

    target_key = "bg"  # for the bandgap benchmark

    # validation data is split from training data
    train_data = load_json_to_df(cfg.DATASET.TRAIN, target_key=target_key)
    test_data = load_json_to_df(cfg.DATASET.TEST, target_key=target_key)

    if args.pretrain:
        val_data = train_data.sample(frac=cfg.DATASET.VAL_FRAC, random_state=cfg.SOLVER.SEED)
        # make train and validation data into lists for k-fold compatibility
        train_data_list = [train_data.drop(val_data.index).reset_index(drop=True)]
        val_data_list = [val_data.reset_index(drop=True)]
    else:
        train_data_list, val_data_list = split_data_by_kfold(
            train_data, n_splits=cfg.SOLVER.NUM_FOLDS, seed=cfg.SOLVER.SEED
        )

    for train_data, val_data in zip(train_data_list, val_data_list):
        ml_models = ["random_forest", "linear_regression", "svr"]
        gnn_models = ["cgcnn", "leftnet", "cartnet"]
        dataset = CrystalDataset(
            train_df=train_data,
            val_df=val_data,
            test_df=test_data,
            cif_folder=cfg.MODEL.CIF_FOLDER,
            init_file=cfg.MODEL.INIT_FILE,
            max_nbrs=cfg.MODEL.MAX_NBRS,
            radius=cfg.MODEL.RADIUS,
            target_key="bg",
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.SOLVER.WORKERS,
            randomize_train=cfg.SOLVER.RANDOMIZE,
        )

        # --- traditional ML branch ---
        if cfg.MODEL.NAME not in ml_models + gnn_models:
            raise ValueError(f"Model {cfg.MODEL.NAME} not recognized.")
        if cfg.MODEL.NAME in ml_models:
            print(f"Training classic model: {cfg.MODEL.NAME}")
            x_train, y_train = extract_features(dataset.train_dataset)
            x_valid, y_valid = extract_features(dataset.valid_dataset)
            x_test, y_test = extract_features(dataset.test_dataset)

            run_traditional_ml(
                model_name=cfg.MODEL.NAME,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                seed=cfg.SOLVER.SEED,
                x_test=x_test,
                y_test=y_test,
            )

        # --- GNNs ---
        elif cfg.MODEL.NAME in gnn_models:
            train_loader = dataset.get_train_loader()
            val_loader = dataset.get_valid_loader()
            test_data = dataset.get_test_loader()

            atom_fea_len, nbr_fea_len, pos_fea_len = dataset.feature_dims()
            model = get_model(cfg, atom_fea_len, nbr_fea_len, pos_fea_len)

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

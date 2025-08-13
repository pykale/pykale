# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
DrugBAN on BindingDB/BioSNAP/Human drug-protein interaction prediction

Reference: https://github.com/peizhenbai/DrugBAN/blob/main/main.py
"""

import argparse
import os
import warnings
from datetime import datetime
from time import time

import pytorch_lightning as pl
import torch
from configs import get_cfg_defaults
from model import get_dataloader, get_dataset, get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from kale.loaddata.molecular_datasets import graph_collate_func
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args


def main():
    # ---- ignore warnings ----
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    SEED = cfg.SOLVER.SEED
    set_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    # ---- setup dataset ----
    data_folder = os.path.join(f"./datasets/{cfg.DATA.DATASET}", str(cfg.DATA.SPLIT))
    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Dataset folder {data_folder} does not exist. Please check if the data folder exists.\n"
            f"If you haven't downloaded the data, please follow the dataset guidance at https://github.com/pykale/pykale/tree/main/examples/bindingdb_drugban#datasets"
        )
    if not cfg.DA.TASK:
        datasets = get_dataset(data_folder, da_task=cfg.DA.TASK)
    else:
        datasets = get_dataset(data_folder, da_task=cfg.DA.TASK)

    # ---- setup dataloader ----
    training_dataloader, valid_dataloader, test_dataloader = get_dataloader(
        *datasets,
        batchsize=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        collate_fn=graph_collate_func,
        is_da=cfg.DA.USE,
        da_task=cfg.DA.TASK,
    )

    # ---- set logger ----
    experiment_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if cfg.COMET.USE:
        logger = pl_loggers.CometLogger(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            save_dir=os.path.join(cfg.OUTPUT.OUT_DIR, "comet"),
            experiment_name=os.path.join(cfg.COMET.EXPERIMENT_NAME, experiment_time),
        )
    else:
        logger = pl_loggers.TensorBoardLogger(
            save_dir=os.path.join(cfg.OUTPUT.OUT_DIR, "tensorboard", cfg.COMET.PROJECT_NAME),
            name=experiment_time,
        )

    # ---- setup trainer ----
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{val_BinaryAUROC:.4f}",
        monitor="val_BinaryAUROC",
        mode="max",
    )

    model = get_model(cfg)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=cfg["SOLVER"]["MAX_EPOCH"],
        logger=logger,
        deterministic=True,  # for reproducibility
    )
    trainer.fit(model, train_dataloaders=training_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path="best")


if __name__ == "__main__":
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

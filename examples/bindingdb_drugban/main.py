# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
DrugBAN on BindingDB/BioSNAP/Human drug-protein interaction prediction

Reference: https://github.com/peizhenbai/DrugBAN/blob/main/main.py
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from time import time

import pytorch_lightning as pl
import torch
from configs import get_cfg_defaults
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

sys.path.append("../../../pykale/")
from model import get_dataloader, get_dataset, get_model
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
    dataFolder = os.path.join(f"./datasets/{cfg.DATA.DATASET}", str(cfg.DATA.SPLIT))
    if not cfg.DA.TASK:
        datasets = get_dataset(dataFolder, da_task=cfg.DA.TASK)
    else:
        datasets = get_dataset(dataFolder, da_task=cfg.DA.TASK)

    # ---- setup dataloader ----
    training_generator, valid_generator, test_generator = get_dataloader(
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
        logger = CometLogger(
            api_key=cfg["COMET"]["API_KEY"],
            project_name=cfg["COMET"]["PROJECT_NAME"],
            experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, experiment_time),
        )
    else:
        logger = TensorBoardLogger(save_dir=cfg.RESULT.OUTPUT_DIR, name=experiment_time)

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
    trainer.fit(model, train_dataloaders=training_generator, val_dataloaders=valid_generator)
    trainer.test(model, dataloaders=test_generator, ckpt_path="best")


if __name__ == "__main__":
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

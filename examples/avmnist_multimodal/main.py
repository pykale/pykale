"""This example is about the application of Multimodal Neural Network (MMNN) for digit classification (0 and 1) on the AVMNIST dataset.

Reference: https://github.com/pliang279/MultiBench/tree/main/examples/multimedia
"""

import argparse
import os
import time

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

from kale.loaddata.avmnist_datasets import AVMNISTDataset
from kale.utils.download import download_file_gdrive
from kale.utils.logger import construct_logger
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="PyTorch AVMNIST Training")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--output", default="default", help="folder to save output", type=str)
    parser.add_argument(
        "--gpus",
        default=0,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # ---- setup logger and output ----
    output_dir = os.path.join(cfg.OUTPUT.OUT_DIR, cfg.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = construct_logger("avmnist", output_dir)
    logger.info("Using " + device)
    logger.info("\n" + cfg.dump())

    download_file_gdrive(cfg.DATASET.GDRIVE_ID, cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.FILE_FORMAT)

    dataset = AVMNISTDataset(data_dir=cfg.DATASET.ROOT, batch_size=cfg.DATASET.BATCH_SIZE)
    traindata = dataset.get_train_loader()
    validdata = dataset.get_valid_loader()
    testdata = dataset.get_test_loader()

    model = get_model(cfg, device)

    # Choose one logger (CometLogger or TensorBoardLogger) using cfg.COMET.ENABLE
    if cfg.COMET.ENABLE:
        suffix = str(int(time.time() * 1000))[6:]
        logger = pl_loggers.CometLogger(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            save_dir=cfg.OUTPUT.OUT_DIR,
            experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, suffix),
        )
    else:
        logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR)

    # ---- setup callbacks ----
    # setup progress bar
    progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

    # setup learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---- setup trainers ----
    trainer = pl.Trainer(
        default_root_dir=cfg.OUTPUT.OUT_DIR,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        accelerator="gpu" if args.gpus != 0 else "cpu",
        logger=logger,
        callbacks=[progress_bar, lr_monitor],
        log_every_n_steps=1,
    )
    # ---- start training ----
    trainer.fit(model, traindata, validdata)

    # ---- start testing ----
    print("Testing:")
    trainer.test(model, testdata)


if __name__ == "__main__":
    main()

"""
This example demonstrates the use of a CNN and a Transformer-Encoder
for image classification on CIFAR10 Dataset, using PyTorch Lightning.
Reference: See kale.embed.attention_cnn for more details.
"""

import argparse

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers

import kale.utils.seed as seed
from kale.loaddata.image_access import get_cifar


def arg_parse():
    parser = argparse.ArgumentParser(description="CNN Transformer on CIFAR10 Dataset")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup config ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed.set_seed(cfg.SOLVER.SEED)

    # ---- setup dataset ----
    train_loader, valid_loader = get_cifar(cfg)

    # ---- setup model ----
    print("==> Building model..")
    model, optim = get_model(cfg)

    # ---- setup logger ----
    logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT_DIR)

    # ---- setup trainers ----
    trainer = pl.Trainer(
        default_root_dir=cfg.OUTPUT_DIR,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        logger=logger,
        accelerator="auto",
        # strategy="ddp",  # not work on Windows, which does not support CCL backend
        log_every_n_steps=1,
    )

    # ---- start training ----
    trainer.fit(model, train_loader, valid_loader)

    # ---- start testing ----
    trainer.test(model, valid_loader)


if __name__ == "__main__":
    main()

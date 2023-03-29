"""
This example demonstrates the use of a CNN and a Transformer-Encoder
for image classification on CIFAR10.
Reference: See kale.embed.attention_cnn for more details.
"""

import argparse

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model

import kale.utils.seed as seed
from kale.loaddata.image_access import get_cifar


def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
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

    # ---- setup trainers ----
    trainer = pl.Trainer(
        default_root_dir=cfg.OUTPUT_DIR,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        accelerator="auto",
        strategy="ddp",
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, train_loader)


if __name__ == "__main__":
    main()

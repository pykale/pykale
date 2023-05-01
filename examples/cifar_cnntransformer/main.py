"""This example is about the use of a CNN and a Transformer-Encoder for image classification on CIFAR10 Dataset,
using PyTorch Lightning.

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., & Polosukhin, I.  (2017).
    Attention Is All You Need.  In Proceedings of the Advances in Neural Information Processing Systems(pp. 6000-6010).
    https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
"""

import argparse
import time

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

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
    model = get_model(cfg)

    # ---- setup logger ----
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
        devices=args.gpus,
        logger=logger,
        callbacks=[progress_bar, lr_monitor],
        strategy="ddp",  # comment this line on Windows, because Windows does not support CCL backend
        log_every_n_steps=1,
    )

    # ---- start training ----
    trainer.fit(model, train_loader, valid_loader)

    # ---- start testing ----
    trainer.test(model, valid_loader)


if __name__ == "__main__":
    main()

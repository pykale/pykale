"""This example is about domain adaptation for digit image datasets, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging
import time

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from kale.loaddata.image_access import DigitDataset
from kale.loaddata.multi_domain import MultiDomainDatasets
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Digits Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    args = parser.parse_args()
    return args


def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup output ----
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)

    # ---- setup dataset ----
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(cfg.DATASET.SOURCE.upper()), DigitDataset(cfg.DATASET.TARGET.upper()), cfg.DATASET.ROOT
    )
    dataset = MultiDomainDatasets(
        source,
        target,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
        valid_split_ratio=cfg.DATASET.VALID_SPLIT_RATIO,
    )

    # Repeat multiple times to get std
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i * 10
        # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        set_seed(seed)
        print(f"==> Building model for seed {seed} ......")

        # ---- setup model ----
        model, train_params = get_model(cfg, dataset, num_channels)

        # ---- setup logger ----
        if cfg.COMET.ENABLE:
            suffix = str(int(time.time() * 1000))[6:]
            logger = pl_loggers.CometLogger(
                api_key=cfg.COMET.API_KEY,
                project_name=cfg.COMET.PROJECT_NAME,
                save_dir=cfg.OUTPUT.OUT_DIR,
                experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, suffix),
            )
        else:
            logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR, name="seed{}".format(seed))

        # ---- setup callbacks ----
        # setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min",
        )

        # setup progress bar
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        # ---- setup trainer ----
        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            accelerator="gpu" if args.devices != 0 else "cpu",
            devices=args.devices,
            callbacks=[checkpoint_callback, progress_bar],
            logger=logger,
        )

        # ---- start training ----
        trainer.fit(model)

        # ---- start testing ----
        trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()

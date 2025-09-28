"""This example is about domain adaptation for digit image datasets, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging
import os
import time

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from kale.loaddata.image_access import DigitDataset, ImageAccess
from kale.loaddata.multi_domain import BinaryDomainDatasets, MultiDomainAccess, MultiDomainDataset
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Digits Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
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

    outdir = os.path.join(
        cfg.OUTPUT.OUT_DIR,
        cfg.DATASET.SOURCE + "2" + cfg.DATASET.TARGET + "_" + cfg.DAN.METHOD,
    )

    # ---- setup dataset ----
    # source, target, num_channels = DigitDataset.get_source_target(
    #     DigitDataset(cfg.DATASET.SOURCE.upper()), DigitDataset(cfg.DATASET.TARGET.upper()), cfg.DATASET.ROOT
    # )
    # dataset = BinaryDomainDatasets(
    #     source,
    #     target,
    #     config_weight_type=cfg.DATASET.WEIGHT_TYPE,
    #     config_size_type=cfg.DATASET.SIZE_TYPE,
    #     valid_split_ratio=cfg.DATASET.VALID_SPLIT_RATIO,
    # )
    data_src = DigitDataset(cfg.DATASET.SOURCE.upper())
    data_tgt = DigitDataset(cfg.DATASET.TARGET.upper())
    num_channels = max(DigitDataset.get_channel_numbers(data_src), DigitDataset.get_channel_numbers(data_tgt))
    data_access = MultiDomainAccess(
        {
            cfg.DATASET.SOURCE.upper(): DigitDataset.get_access(data_src, cfg.DATASET.ROOT),
            cfg.DATASET.TARGET.upper(): DigitDataset.get_access(data_tgt, cfg.DATASET.ROOT),
        },
        cfg.DATASET.NUN_CLASSES,
        return_domain_label=True,
    )
    # data_access = ImageAccess.get_multi_domain_images(
    #     "DIGITS",
    #     cfg.DATASET.ROOT,
    #     sub_domain_set=[cfg.DATASET.SOURCE.upper(), cfg.DATASET.TARGET.upper()],
    #     return_domain_label=True,
    # )
    dataset = MultiDomainDataset(data_access)

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
                save_dir=outdir,
                experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, suffix),
            )
        else:
            logger = pl_loggers.TensorBoardLogger(outdir, name="seed{}".format(seed))

        # ---- setup callbacks ----
        # setup checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            monitor="valid_loss",
            mode="min",
        )

        # setup progress bar
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        # ---- setup trainer ----
        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            accelerator="gpu" if args.devices != 0 else "cpu",
            devices=args.devices if args.devices != 0 else "auto",
            callbacks=[checkpoint_callback, progress_bar],
            logger=logger,
            log_every_n_steps=cfg.SOLVER.LOG_EVERY_N_STEPS,
        )

        # ---- start training ----
        trainer.fit(model)

        # ---- start testing ----
        trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()

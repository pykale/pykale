"""This example is about domain adaptation for digit image datasets, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging
import os

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model

from kale.loaddata.digits_access import DigitDataset
from kale.loaddata.multi_domain import MultiDomainDatasets
from kale.utils.csv_logger import setup_logger
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Digits Datasets")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--gpus", default="0", help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
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
    os.makedirs(cfg.OUTPUT.DIR, exist_ok=True)
    format_str = "@%(asctime)s %(name)s [%(levelname)s] - (%(message)s)"
    logging.basicConfig(format=format_str)
    # ---- setup dataset ----
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(cfg.DATASET.SOURCE.upper()), DigitDataset(cfg.DATASET.TARGET.upper()), cfg.DATASET.ROOT
    )
    dataset = MultiDomainDatasets(
        source, target, config_weight_type=cfg.DATASET.WEIGHT_TYPE, config_size_type=cfg.DATASET.SIZE_TYPE
    )

    # Repeat multiple times to get std
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i * 10
        # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        set_seed(seed)
        print(f"==> Building model for seed {seed} ......")
        # ---- setup model and logger ----
        model, train_params = get_model(cfg, dataset, num_channels)
        logger, results, checkpoint_callback, test_csv_file = setup_logger(
            train_params, cfg.OUTPUT.DIR, cfg.DAN.METHOD, seed
        )
        trainer = pl.Trainer(
            progress_bar_refresh_rate=cfg.OUTPUT.PB_FRESH,  # in steps
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            checkpoint_callback=checkpoint_callback,
            # resume_from_checkpoint=last_checkpoint_file,
            gpus=args.gpus,
            logger=False,  # logger,
            # weights_summary='full',
            fast_dev_run=True,  # True,
        )

        trainer.fit(model)
        results.update(
            is_validation=True, method_name=cfg.DAN.METHOD, seed=seed, metric_values=trainer.callback_metrics,
        )
        # test scores
        trainer.test()
        results.update(
            is_validation=False, method_name=cfg.DAN.METHOD, seed=seed, metric_values=trainer.callback_metrics,
        )
        results.to_csv(test_csv_file)
        results.print_scores(cfg.DAN.METHOD)


if __name__ == "__main__":
    main()

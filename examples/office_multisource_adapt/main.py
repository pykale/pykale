"""This example is about domain adaptation for digit image datasets, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from kale.loaddata.image_access import ImageAccess
from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Multi-source domain adaptation")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    """The main for this multi-source domain adaptation example, showing the workflow"""
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
    if type(cfg.DATASET.SOURCE) == list:
        sub_domain_set = cfg.DATASET.SOURCE + [cfg.DATASET.TARGET]
    else:
        sub_domain_set = None
    num_channels = cfg.DATASET.NUM_CHANNELS
    if cfg.DATASET.NAME.upper() == "DIGITS":
        kwargs = {"return_domain_label": True}
    else:
        kwargs = {"download": True, "return_domain_label": True}

    data_access = ImageAccess.get_multi_domain_images(
        cfg.DATASET.NAME.upper(), cfg.DATASET.ROOT, sub_domain_set=sub_domain_set, **kwargs
    )

    # Repeat multiple times to get std
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = cfg.SOLVER.SEED + i * 10
        dataset = MultiDomainAdapDataset(data_access, random_state=seed)
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")
        # ---- setup model and logger ----
        model, train_params = get_model(cfg, dataset, num_channels)

        tb_logger = TensorBoardLogger(cfg.OUTPUT.TB_DIR, name="seed{}".format(seed))
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min",
        )
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            callbacks=[checkpoint_callback, progress_bar],
            gpus=args.gpus,
            auto_select_gpus=True,
            logger=tb_logger,  # logger,
            # weights_summary='full',
            fast_dev_run=False,  # True,
        )

        trainer.fit(model)
        trainer.test()


if __name__ == "__main__":
    main()

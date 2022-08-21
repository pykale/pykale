"""This example is about domain adaptation for action recognition, using PyTorch Lightning.

Reference: https://github.com/thuml/CDAN/blob/master/pytorch/train_image.py
"""

import argparse
import logging

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.utils.seed import set_seed

# from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Domain Adversarial Networks on Action Datasets")
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
    seed = cfg.SOLVER.SEED
    source, target, num_classes = VideoDataset.get_source_target(
        VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    )
    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        seed=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
    )

    # ---- training/test process ----
    ### Repeat multiple times to get std
    for i in range(0, cfg.DATASET.NUM_REPEAT):
        seed = seed + i * 10
        set_seed(seed)  # seed_everything in pytorch_lightning did not set torch.backends.cudnn
        print(f"==> Building model for seed {seed} ......")
        # ---- setup model and logger ----
        model, train_params = get_model(cfg, dataset, num_classes)
        tb_logger = pl_loggers.TensorBoardLogger(cfg.OUTPUT.TB_DIR, name="seed{}".format(seed))
        checkpoint_callback = ModelCheckpoint(
            # dirpath=full_checkpoint_dir,
            filename="{epoch}-{step}-{valid_loss:.4f}",
            # save_last=True,
            # save_top_k=1,
            monitor="valid_loss",
            mode="min",
        )

        ### Set early stopping
        # early_stop_callback = EarlyStopping(monitor="valid_target_acc", min_delta=0.0000, patience=100, mode="max")

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        progress_bar = TQDMProgressBar(cfg.OUTPUT.PB_FRESH)

        ### Set the lightning trainer. Comment `limit_train_batches`, `limit_val_batches`, `limit_test_batches` when
        # training. Uncomment and change the ratio to test the code on the smallest sub-dataset for efficiency in
        # debugging. Uncomment early_stop_callback to activate early stopping.
        trainer = pl.Trainer(
            min_epochs=cfg.SOLVER.MIN_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            # resume_from_checkpoint=last_checkpoint_file,
            gpus=args.gpus,
            logger=tb_logger,  # logger,
            # weights_summary='full',
            fast_dev_run=cfg.OUTPUT.FAST_DEV_RUN,  # True,
            callbacks=[lr_monitor, checkpoint_callback, progress_bar],
            # callbacks=[early_stop_callback, lr_monitor],
            # limit_train_batches=0.005,
            # limit_val_batches=0.06,
            # limit_test_batches=0.06,
        )

        ### Find learning_rate
        # lr_finder = trainer.tuner.lr_find(model, max_lr=0.1, min_lr=1e-6)
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        # logging.info(lr_finder.suggestion())

        ### Training/validation process
        trainer.fit(model)

        ### Test process
        trainer.test()


if __name__ == "__main__":
    main()

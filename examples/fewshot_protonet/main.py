"""
This demo trains a prototypical network to solve N-Way-K-Shot problems.

- N-way: This refers to the number of different classes or categories involved in evaluation. For example, in a 5-way problem, the model is presented with instances from 5 different classes.

- K-shot: This indicates the number of examples (or "shots") from each class that the model has access to in training and evaluation. In a 1-shot learning task, the model gets only one example per class, while in a 3-shot task, it gets three examples per class.

By default, this example uses the Omniglot dataset, which can be downloaded from https://github.com/brendenlake/omniglot.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017. Prototypical Networks for Few-shot Learning. Advances in Neural Information Processing Systems, 30.
"""
import argparse
import os
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
from config import get_cfg_defaults
from torch.utils.data import DataLoader
from torchvision.models import *

from kale.embed.image_cnn import *
from kale.loaddata.few_shot import NWayKShotDataset
from kale.pipeline.fewshot_trainer import ProtoNetTrainer
from kale.prepdata.image_transform import get_transform


def arg_parse():
    parser = argparse.ArgumentParser(description="Args of ProtoNet")
    parser.add_argument("--cfg", default="configs/demo.yaml", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--ckpt", default=None, type=Optional[str])
    args = parser.parse_args()
    return args


def main():
    # ---- get args ----
    args = arg_parse()

    # ---- get configurations ----
    cfg_path = args.cfg
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    # ---- set model ----
    net = eval(f"{cfg.MODEL.BACKBONE}(weights={cfg.MODEL.PRETRAIN_WEIGHTS})")
    if cfg.MODEL.BACKBONE.startswith("resnet"):
        net.fc = Flatten()

    model = ProtoNetTrainer(
        net=net,
        train_num_classes=cfg.TRAIN.NUM_CLASSES,
        train_num_support_samples=cfg.TRAIN.NUM_SUPPORT_SAMPLES,
        train_num_query_samples=cfg.TRAIN.NUM_QUERY_SAMPLES,
        val_num_classes=cfg.VAL.NUM_CLASSES,
        val_num_support_samples=cfg.VAL.NUM_SUPPORT_SAMPLES,
        val_num_query_samples=cfg.VAL.NUM_QUERY_SAMPLES,
        devices=cfg.DEVICE,
        optimizer=cfg.TRAIN.OPTIMIZER,
        lr=cfg.TRAIN.LEARNING_RATE,
    )

    # ---- set data loader ----
    transform = get_transform(kind="few-shot", augment=False)
    train_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode="train",
        num_support_samples=cfg.TRAIN.NUM_SUPPORT_SAMPLES,
        num_query_samples=cfg.TRAIN.NUM_QUERY_SAMPLES,
        transform=transform,
    )
    train_dataloader = DataLoader(train_set, batch_size=cfg.TRAIN.NUM_CLASSES, shuffle=True, drop_last=True)
    val_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode="val",
        num_support_samples=cfg.VAL.NUM_SUPPORT_SAMPLES,
        num_query_samples=cfg.VAL.NUM_QUERY_SAMPLES,
        transform=transform,
    )
    val_dataloader = DataLoader(val_set, batch_size=cfg.VAL.NUM_CLASSES, drop_last=True)
    test_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode="test",
        num_support_samples=cfg.VAL.NUM_SUPPORT_SAMPLES,
        num_query_samples=cfg.VAL.NUM_QUERY_SAMPLES,
        transform=transform,
    )
    test_dataloader = DataLoader(test_set, batch_size=cfg.VAL.NUM_CLASSES, drop_last=True)

    # ---- set logger ----
    experiment_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = pl.loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR, name=experiment_time)
    logger.log_hyperparams(cfg)

    # ---- set callbacks ----
    dirpath = os.path.join(cfg.OUTPUT.OUT_DIR, experiment_time, cfg.OUTPUT.WEIGHT_DIR)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=cfg.OUTPUT.SAVE_TOP_K,
        save_last=cfg.OUTPUT.SAVE_LAST,
        verbose=True,
    )

    # ---- set trainer ----
    trainer = pl.Trainer(
        devices=args.devices,
        max_epochs=cfg.TRAIN.EPOCHS,
        logger=logger,
        callbacks=[model_checkpoint],
        accelerator="gpu" if args.devices > 0 else "cpu",
        log_every_n_steps=cfg.OUTPUT.SAVE_FREQ,
    )

    # ---- training ----
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.ckpt)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path="best")


if __name__ == "__main__":
    main()

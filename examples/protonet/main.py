"""
This example is about training prototypical networks to perform N-Way-K-Shot problems.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30.
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import *

sys.path.append("/home/wenrui/Projects/pykale/")
from kale.embed.image_cnn import *
from kale.loaddata.few_shot import NWayKShotDataset
from kale.pipeline.protonet import ProtoNetTrainer


def arg_parse():
    parser = argparse.ArgumentParser(description="Args of ProtoNet")
    parser.add_argument("--cfg", default="configs/omniglot_resnet18_5way5shot.yaml", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--ckpt", default=None, type=Any)
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
        train_n_way=cfg.TRAIN.N_WAYS,
        train_k_shot=cfg.TRAIN.K_SHOTS,
        train_k_query=cfg.TRAIN.K_QUERIES,
        val_n_way=cfg.VAL.N_WAYS,
        val_k_shot=cfg.VAL.K_SHOTS,
        val_k_query=cfg.VAL.K_QUERIES,
        devices=cfg.DEVICE,
        optimizer=cfg.TRAIN.OPTIMIZER,
        lr=cfg.TRAIN.LEARNING_RATE,
    )

    # ---- set data loader ----
    transform = transforms.Compose(
        [transforms.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)), transforms.ToTensor()]
    )
    train_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode="train",
        k_shot=cfg.TRAIN.K_SHOTS,
        query_samples=cfg.TRAIN.K_QUERIES,
        transform=transform,
    )
    train_dataloader = DataLoader(train_set, batch_size=cfg.TRAIN.N_WAYS, shuffle=True, drop_last=True)
    val_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT, mode="val", k_shot=cfg.VAL.K_SHOTS, query_samples=cfg.VAL.K_QUERIES, transform=transform
    )
    val_dataloader = DataLoader(val_set, batch_size=cfg.VAL.N_WAYS, drop_last=True)
    test_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT, mode="test", k_shot=cfg.VAL.K_SHOTS, query_samples=cfg.VAL.K_QUERIES, transform=transform
    )
    test_dataloader = DataLoader(test_set, batch_size=cfg.VAL.N_WAYS, drop_last=True)

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

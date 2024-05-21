"""
This example tests the performance of prototypical networks under N-Way-K-Shot settings on unseen data.
Users can apply this script to evaluate their trained models on unseen data without re-training.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017. 
    Prototypical Networks for Few-shot Learning. 
    Advances in Neural Information Processing Systems, 30.
"""
import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import *

from kale.embed.image_cnn import *
from kale.loaddata.few_shot import NWayKShotDataset
from kale.pipeline.fewshot_trainer import ProtoNetTrainer
from kale.prepdata.image_transform import get_transform


def arg_parse():
    parser = argparse.ArgumentParser(description="Args of ProtoNet")
    parser.add_argument("--cfg", default="configs/omniglot_resnet18_5way5shot.yaml", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--ckpt", default=None)
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
    model = ProtoNetTrainer(cfg=cfg, net=net)

    # ---- set data loader ----
    transform = get_transform(kind="few-shot", augment=False)

    test_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT, mode="test", k_shot=cfg.VAL.K_SHOTS, query_samples=cfg.VAL.K_QUERIES, transform=transform
    )
    test_dataloader = DataLoader(test_set, batch_size=cfg.VAL.N_WAYS, drop_last=True)  # must be True

    # ---- set logger ----
    experiment_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = pl.loggers.TensorBoardLogger(cfg.OUTPUT.LOG_DIR, name=experiment_time)
    logger.log_hyperparams(cfg)

    # ---- set callbacks ----
    dirpath = os.path.join(cfg.OUTPUT.LOG_DIR, experiment_time, cfg.OUTPUT.WEIGHT_DIR)
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=cfg.OUTPUT.SAVE_TOP_K,
        save_last=cfg.OUTPUT.SAVE_LAST,
        verbose=True
    )

    # ---- set trainer ----
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=cfg.TRAIN.EPOCHS,
        logger=logger,
        callbacks=[model_checkpoint],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        log_every_n_steps=cfg.OUTPUT.SAVE_FREQ
    )

    # ---- test ----
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()

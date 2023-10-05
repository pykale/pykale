"""
This example is about training prototypical networks to perform N-Way-K-Shot problems.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017.
    Prototypical networks for few-shot learning.
    Advances in neural information processing systems, 30.
"""
import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
from config import get_cfg_defaults
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import *

from kale.embed.image_cnn import *
from kale.loaddata.n_way_k_shot import NWayKShotDataset
from kale.pipeline.protonet import ProtoNetTrainer


def get_parser():
    parser = argparse.ArgumentParser(description="Args of ProtoNet")
    parser.add_argument("--cfg", default="examples/protonet/configs/omniglot_resnet18_5way5shot.yaml", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    return parser


def main():
    # ---- get args ----
    parser = get_parser()
    args = parser.parse_args()

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
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT,
        mode="train",
        k_shot=cfg.TRAIN.K_SHOTS,
        query_samples=cfg.TRAIN.K_QUERIES,
        transform=transform,
    )
    train_dataloader = DataLoader(train_set, batch_size=cfg.TRAIN.N_WAYS, shuffle=True, num_workers=30, drop_last=True)
    val_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT, mode="val", k_shot=cfg.VAL.K_SHOTS, query_samples=cfg.VAL.K_QUERIES, transform=transform
    )
    val_dataloader = DataLoader(val_set, batch_size=cfg.VAL.N_WAYS, num_workers=30, drop_last=True)

    # ---- set logger ----
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = pl.loggers.TensorBoardLogger(cfg.OUTPUT.OUT_DIR, name=dt_string)
    logger.log_hyperparams(cfg)

    # ---- set callbacks ----
    dirpath = os.path.join(cfg.OUTPUT.OUT_DIR, dt_string, cfg.OUTPUT.WEIGHT_DIR)
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
        gpus=args.gpus,
        max_epochs=cfg.TRAIN.EPOCHS,
        logger=logger,
        callbacks=[model_checkpoint],
        accelerator="gpu" if args.gpus > 0 else "cpu",
        log_every_n_steps=cfg.OUTPUT.SAVE_FREQ,
    )

    # ---- training ----
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()

"""
This example is about testing prototypical networks to perform N-Way-K-Shot problems.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017.
    Prototypical networks for few-shot learning.
    Advances in neural information processing systems, 30.
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
from kale.loaddata.n_way_k_shot import NWayKShotDataset
from kale.pipeline.protonet import ProtoNetTrainer


def get_parser():
    parser = argparse.ArgumentParser(description="ProtoNet")
    parser.add_argument("--cfg", default="examples/protonet/configs/omniglot_resnet18_5way5shot.yaml", type=str)
    parser.add_argument("--ckpt", default="examples/protonet/logs/2023-09-26-15-14-41/weights/last.ckpt", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    return parser


def weights_update(model, checkpoint):
    """Load the pre-trained parameters to the model."""
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def main():
    # ---- get args ----
    parser = get_parser()
    args = parser.parse_args()
    args.gpus = min(args.gpus, torch.cuda.device_count())

    # ---- get configurations ----
    cfg_path = args.cfg
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    # ---- set model ----
    net = eval(f"{cfg.MODEL.BACKBONE}(weights={cfg.MODEL.PRETRAIN_WEIGHTS})")
    if cfg.MODEL.BACKBONE.startswith("resnet"):
        net.fc = Flatten()
    model = ProtoNetTrainer(cfg=cfg, net=net)

    # ---- set data loader ----
    transform = transforms.Compose(
        [transforms.Resize((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE)), transforms.ToTensor()]
    )

    test_set = NWayKShotDataset(
        path=cfg.DATASET.ROOT, mode="test", k_shot=cfg.VAL.K_SHOTS, query_samples=cfg.VAL.K_QUERIES, transform=transform
    )
    test_dataloader = DataLoader(test_set, batch_size=cfg.VAL.N_WAYS, num_workers=30, drop_last=True)  # must be True

    # ---- set logger ----
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = pl.loggers.TensorBoardLogger(cfg.OUTPUT.LOG_DIR, name=dt_string)
    logger.log_hyperparams(cfg)

    # ---- set callbacks ----
    dirpath = os.path.join(cfg.OUTPUT.LOG_DIR, dt_string, cfg.OUTPUT.WEIGHT_DIR)
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
        resume_from_checkpoint=args.ckpt,
    )

    # ---- test ----
    model_test = weights_update(model=model, checkpoint=torch.load(args.ckpt))
    trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()

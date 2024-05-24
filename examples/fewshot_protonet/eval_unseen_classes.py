"""
This demo tests the performance of a pretrained prototypical network on unseen classes.
Users can apply this script to evaluate their pretrained models on unseen classes without re-training.

Reference:
    Snell, J., Swersky, K. and Zemel, R., 2017. Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems, 30.
"""
import argparse
from datetime import datetime

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
    parser.add_argument("--cfg", default="configs/demo.yaml", type=str, help="Path to the configuration file")
    parser.add_argument("--ckpt", default=None, help="Path to the checkpoint file")
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
    print(f"Using backbone: {cfg.MODEL.BACKBONE}")
    net = eval(f"{cfg.MODEL.BACKBONE}(weights={cfg.MODEL.PRETRAIN_WEIGHTS})")
    if cfg.MODEL.BACKBONE.startswith("resnet"):
        net.fc = Flatten()
    model = ProtoNetTrainer(net=net, devices="cuda" if cfg.GPUS > 0 else "cpu")

    # ---- set data loader ----
    transform = get_transform(kind="few-shot", augment=False)
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

    # ---- set trainer ----
    trainer = pl.Trainer(
        devices=cfg.GPUS,
        max_epochs=cfg.TRAIN.EPOCHS,
        logger=logger,
        accelerator="gpu" if cfg.GPUS > 0 else "cpu",
        log_every_n_steps=cfg.OUTPUT.SAVE_FREQ,
    )

    # ---- testing on unseen classes ----
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()

import argparse

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from kale.loaddata.tdc_datasets import BindingDBDataset


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DeepDTA on BindingDB dataset")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--gpus", default="0", help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- set configs, logger and device ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    tb_logger = TensorBoardLogger("tb_logs", name=cfg.DATASET.NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- set dataset ----
    train_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="train", path=cfg.DATASET.PATH)
    val_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="valid", path=cfg.DATASET.PATH)
    test_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="test", path=cfg.DATASET.PATH)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)

    # ---- set model ----
    model = get_model(cfg)

    # ---- training and evaluation ----
    gpus = 1 if device == "cuda" else 0
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(max_epochs=cfg.SOLVER.MAX_EPOCHS, gpus=gpus, logger=tb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

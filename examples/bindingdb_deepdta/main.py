import argparse

import pytorch_lightning as pl
from config import get_cfg_defaults
from model import get_model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from kale.loaddata.tdc_datasets import BindingDBDataset


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DeepDTA on BindingDB dataset")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- set configs, logger and device ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    tb_logger = TensorBoardLogger("outputs", name=cfg.DATASET.NAME)

    # ---- set dataset ----
    train_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="train", path=cfg.DATASET.PATH)
    valid_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="valid", path=cfg.DATASET.PATH)
    test_dataset = BindingDBDataset(name=cfg.DATASET.NAME, split="test", path=cfg.DATASET.PATH)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)

    # ---- set model ----
    model = get_model(cfg)

    # ---- training and evaluation ----
    checkpoint_callback = ModelCheckpoint(filename="{epoch}-{step}-{valid_loss:.4f}", monitor="valid_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        accelerator="gpu" if args.devices != 0 else "cpu",
        devices=args.devices if args.devices != 0 else "auto",
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(dataloaders=test_loader)


if __name__ == "__main__":
    main()

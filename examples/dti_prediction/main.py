import argparse

import pytorch_lightning as pl
from config import get_cfg_defaults
from dataset import DTIDeepDataset
from model import get_model
from torch.utils.data import DataLoader


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='DeepDTA on BindingDB dataset')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--gpus', default='0', help='gpu id(s) to use', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- set configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ---- set dataset ----
    train_dataset = DTIDeepDataset(dataset=cfg.DATASET.NAME, split="train")
    val_dataset = DTIDeepDataset(dataset=cfg.DATASET.NAME, split="valid")
    test_dataset = DTIDeepDataset(dataset=cfg.DATASET.NAME, split="test")
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=cfg.SOLVER.TEST_BATCH_SIZE)

    # ---- set model ----
    model = get_model(cfg)

    # ---- training and evaluation ----
    trainer = pl.Trainer(max_epochs=100, gpus=1)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

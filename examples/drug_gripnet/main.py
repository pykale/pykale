import argparse
import os

import torch
from config import get_cfg_defaults
from loaddata import construct_dataset
from model import GripNet
from trainer import Trainer

import kale.utils.logger as lu
import kale.utils.seed as seed


def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--output", default="default", help="folder to save output", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed.set_seed(cfg.SOLVER.SEED)
    # ---- setup logger and output ----
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = lu.construct_logger("gripnet", output_dir)
    logger.info("Using " + device)
    logger.info(cfg.dump())
    # ---- setup dataset ----
    data = construct_dataset(cfg)
    device = torch.device(device)
    data = data.to(device)
    # ---- setup model ----
    print("==> Building model..")
    model = GripNet(
        cfg.GRIPN.GG_LAYERS, cfg.GRIPN.GD_LAYERS, cfg.GRIPN.DD_LAYERS, data.n_d_node, data.n_g_node, data.n_dd_edge_type
    ).to(device)
    # TODO Visualize model
    # ---- setup trainers ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    # TODO
    trainer = Trainer(cfg, device, data, model, optimizer, logger, output_dir)

    if args.resume:
        # Load checkpoint
        print("==> Resuming from checkpoint..")
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp["net"])
        trainer.optim.load_state_dict(cp["optim"])
        trainer.epochs = cp["epoch"]
        trainer.train_auprc = cp["train_auprc"]
        trainer.val_auprc = cp["val_auprc"]
        trainer.train_auroc = cp["train_auroc"]
        trainer.val_auroc = cp["val_auroc"]
        trainer.train_ap = cp["train_ap"]
        trainer.val_ap = cp["val_ap"]

    trainer.train()


if __name__ == "__main__":
    main()

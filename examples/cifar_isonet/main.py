"""ISONet (an extension of ResNet) on CIFAR image classification

Reference: https://github.com/HaozhiQi/ISONet/blob/master/train.py
"""

import argparse
import os

import torch
from config import get_cfg_defaults
from model import get_model
from trainer import Trainer

from kale.loaddata.cifar_access import get_cifar
from kale.utils.logger import construct_logger
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--output", default="default", help="folder to save output", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    """The main for this domain adapation example, showing the workflow"""
    args = arg_parse()

    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # ---- setup logger and output ----
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = construct_logger("isonet", output_dir)
    logger.info("Using " + device)
    logger.info("\n" + cfg.dump())

    # ---- setup dataset ----
    train_loader, val_loader = get_cifar(cfg)

    print("==> Building model..")
    net = get_model(cfg)
    # print(net)
    net = net.to(device)
    # model_stats = summary(net, (3, 32, 32))
    # logger.info('\n'+str(model_stats))

    # Needed even for single GPU https://discuss.pytorch.org/t/attributeerror-net-object-has-no-attribute-module/45652
    if device == "cuda":
        net = torch.nn.DataParallel(net)

    optim = torch.optim.SGD(
        net.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        dampening=cfg.SOLVER.DAMPENING,
        nesterov=cfg.SOLVER.NESTEROV,
    )

    trainer = Trainer(device, train_loader, val_loader, net, optim, logger, output_dir, cfg)

    if args.resume:
        # Load checkpoint
        print("==> Resuming from checkpoint..")
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp["net"])
        trainer.optim.load_state_dict(cp["optim"])
        trainer.epochs = cp["epoch"]
        trainer.train_acc = cp["train_accuracy"]
        trainer.val_acc = cp["test_accuracy"]

    trainer.train()


if __name__ == "__main__":
    main()

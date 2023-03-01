"""
This example demonstrates the use of a CNN and a Transformer-Encoder
for image classification on CIFAR10.

Reference: See kale.embed.attention_cnn for more details.
"""
import argparse
import os

import torch
from config import get_cfg_defaults
from model import get_model
from torchsummary import summary
from trainer import CNNTrainer
import pytorch_lightning as pl

import kale.utils.logger as logging
import kale.utils.seed as seed
from kale.loaddata.image_access import get_cifar


def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    # parser.add_argument('--output', default='default', help='folder to save output', type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
             "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    # ---- setup config ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed.set_seed(cfg.SOLVER.SEED)

    # ---- setup logger ----
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = logging.construct_logger("context_cnns", cfg.OUTPUT_DIR)
    logger.info(f"Using {device}")
    logger.info("\n" + cfg.dump())

    # ---- setup dataset ----
    train_loader, valid_loader = get_cifar(cfg)

    # ---- setup model ----
    print("==> Building model..")
    net = get_model(cfg)
    net = net.to(device)

    model_stats = summary(net, (3, 32, 32))
    logger.info("\n" + str(model_stats))

    if device == "cuda":
        net = torch.nn.DataParallel(net)

    # ---- setup trainers ----
    optim = torch.optim.SGD(
        net.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    
    model = CNNTrainer(train_loader=train_loader, valid_loader=valid_loader, model=net, optim=optim.state_dict(), cfg=cfg)
    trainer = pl.Trainer(
        default_root_dir=cfg.OUTPUT_DIR,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        #accelerator='gpu', devices=1,
        gpus=args.gpus
    )
    if args.resume:
        # Load checkpoint
        print("==> Resuming from checkpoint..")
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp["net"])
        trainer.optim.load_state_dict(cp["optim"])
        trainer.epochs = cp["epoch"]
        trainer.train_acc = cp["train_accuracy"]
        trainer.valid_acc = cp["test_accuracy"]

    trainer.fit(model,train_loader, valid_loader)
    trainer.test(model, train_loader)


if __name__ == "__main__":
    main()

import torch

import argparse
import os
from kale.embed.multimodal_common_fusions import Concat, MultiplicativeInteractions2Modal, LowRankTensorFusion
from kale.embed.lenet import LeNet
from kale.predict.two_layered_mlp import MLP
from kale.loaddata.avmnist_datasets import AVMNISTDataset
from trainer import Trainer

from config import get_cfg_defaults
from kale.utils.seed import set_seed
from kale.utils.logger import construct_logger


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="PyTorch AVMNIST Training")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--output", default="default", help="folder to save output", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args

def main():
    """The main for this avmnist example, showing the workflow"""
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
    logger = construct_logger("avmnist", output_dir)
    logger.info("Using " + device)
    logger.info("\n" + cfg.dump())

    dataset = AVMNISTDataset(cfg.DATASET.ROOT,cfg.DATASET.BATCH_SIZE)
    traindata = dataset.get_train_loader()
    validdata = dataset.get_valid_loader()
    testdata = dataset.get_test_loader()
    print("Data Loaded Successfully")

    encoders = [LeNet(cfg.MODEL.LENET_IN_CHANNELS, cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_IMG), LeNet(cfg.MODEL.LENET_IN_CHANNELS,  cfg.MODEL.CHANNELS, cfg.MODEL.LENET_ADD_LAYERS_AUD)]
    head = MLP(cfg.MODEL.MLP_IN_DIM, cfg.MODEL.MLP_HIDDEN_DIM, cfg.MODEL.OUT_DIM)

    if(cfg.MODEL.FUSION=="late"):
        fusion = Concat()
    elif(cfg.MODEL.FUSION=="tesnor_matrix"):
        fusion = MultiplicativeInteractions2Modal([channels * 8, channels * 32], channels * 40, 'matrix')
    elif(cfg.MODEL.FUSION=="low_rank_tensor"):
        fusion = LowRankTensorFusion([channels * 8, channels * 32], channels * 20, 40)

    #Optimizer args

    trainer = Trainer(encoders, fusion, head, traindata, validdata,testdata, cfg.SOLVER.MAX_EPOCHS, optimtype= torch.optim.SGD, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    print("Model loading succesfully")
    trainer.train()

    print("Testing:")
    model = torch.load('best.pt')  # .cuda()
    trainer.single_test(model)



if __name__ == '__main__':
    main()

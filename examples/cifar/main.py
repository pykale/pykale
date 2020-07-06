# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/train.py 
# Under the MIT License
import os
import argparse
import warnings
import sys
# No need if pykale is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import kale
import torch
from torchsummary import summary

from  config import C
import loaddata as du
# import optim as ou
import kale.utils.logger as lu
import kale.utils.seed as seed
import isonet as isonet
from trainer import Trainer

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='default', help='folder to save output', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    # ---- setup device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using device ' + device)

    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.freeze()
    seed.set_seed(C.SOLVER.SEED)
    # ---- setup logger and output ----
    output_dir = os.path.join(C.OUTPUT_DIR, C.DATASET.NAME, args.output)
    os.makedirs(output_dir, exist_ok=True)
    logger = lu.construct_logger('isonet', output_dir)
    logger.info('Using ' + device)
    logger.info(C.dump())    
    # ---- setup dataset ----
    train_loader, val_loader = du.construct_dataset()

    print('==> Building model..')
    net = isonet.ISONet()
    # print(net)
    net = net.to(device)
    # summary(net, (3, 32, 32))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        # Set to true will be faster but results will vary a bit (not 100% reproducible)
        # torch.backends.cudnn.benchmark = True 

    # optim = ou.construct_optim(net)
    optim = torch.optim.SGD(net.parameters(), lr=C.SOLVER.BASE_LR, momentum=C.SOLVER.MOMENTUM,
                    weight_decay=C.SOLVER.WEIGHT_DECAY, dampening=C.SOLVER.DAMPENING,
                    nesterov=C.SOLVER.NESTEROV)

    trainer = Trainer(
        device,
        train_loader,
        val_loader,
        net,
        optim,
        logger,
        output_dir,
    )

    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        cp = torch.load(args.resume)
        trainer.model.load_state_dict(cp['net'])
        trainer.optim.load_state_dict(cp['optim'])
        trainer.epochs = cp['epoch']
        trainer.train_acc = cp['train_accuracy']
        trainer.val_acc = cp['test_accuracy']

    trainer.train()


if __name__ == '__main__':
    main()

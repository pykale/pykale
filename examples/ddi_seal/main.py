import argparse
import os
import sys

import torch
from config import get_cfg_defaults
from kale.utils.seed import set_seed
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from dataloader import SEALDataset, SEALDynamicDataset
from torch_geometric.data import DataLoader


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--output', default='results', help='folder to save output', type=str)
    parser.add_argument('--data_appendix', type=str, default='')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Using device ' + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # ---- dataset path ----
    if args.data_appendix == '':
        args.data_appendix = '_h{}_{}_rph{}'.format(
            cfg.DATASET.NUM_HOPS, cfg.DATASET.NODE_LABEL, ''.join(str(cfg.DATASET.RATIO_PER_HOP).split('.')))
        if cfg.DATASET.MAX_NODES_PER_HOP:
            args.data_appendix += '_mnph{}'.format(cfg.DATASET.MAX_NODES_PER_HOP)
    path = cfg.DATASET.ROOT + '_seal{}'.format(args.data_appendix)
    print(path)

    # ---- load ogbl data ----
    dataset = PygLinkPropPredDataset(name=cfg.DATASET.NAME)
    split_edge = dataset.get_edge_split()
    data = dataset[0]

    # ---- create dataset ----
    train_dataset = SEALDataset(
        path,
        data,
        split_edge,
        num_hops=cfg.DATASET.NUM_HOPS,
        percent=cfg.DATASET.TRAIN_PERCENT,
        split='train',
        use_coalesce=cfg.DATASET.COALESCE,
        node_label=cfg.DATASET.NODE_LABEL,
        ratio_per_hop=cfg.DATASET.RATIO_PER_HOP,
        max_nodes_per_hop=cfg.DATASET.MAX_NODES_PER_HOP,
    )

    val_dataset = SEALDynamicDataset(
        path,
        data,
        split_edge,
        num_hops=cfg.DATASET.NUM_HOPS,
        percent=cfg.DATASET.VAL_PERCENT,
        split='valid',
        use_coalesce=cfg.DATASET.COALESCE,
        node_label=cfg.DATASET.NODE_LABEL,
        ratio_per_hop=cfg.DATASET.RATIO_PER_HOP,
        max_nodes_per_hop=cfg.DATASET.MAX_NODES_PER_HOP,
    )

    test_dataset = SEALDynamicDataset(
        path,
        data,
        split_edge,
        num_hops=cfg.DATASET.NUM_HOPS,
        percent=cfg.DATASET.TEST_PERCENT,
        split='test',
        use_coalesce=cfg.DATASET.COALESCE,
        node_label=cfg.DATASET.NODE_LABEL,
        ratio_per_hop=cfg.DATASET.RATIO_PER_HOP,
        max_nodes_per_hop=cfg.DATASET.MAX_NODES_PER_HOP,
    )

    max_z = 1000  # set a large max_z so that every z has embeddings to look up

    train_loader = DataLoader(train_dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                              shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                            num_workers=cfg.DATASET.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATASET.BATCH_SIZE,
                             num_workers=cfg.DATASET.NUM_WORKERS)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train, val, test = main()

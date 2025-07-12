# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk
# =============================================================================

"""
Test DrugBAN on BindingDB/BioSNAP/Human drug-protein interaction prediction
"""

import argparse
import os
import warnings
from time import time

import pytorch_lightning as pl
import torch
from configs import get_cfg_defaults

from examples.bindingdb_drugban.model import get_model_from_ckpt, get_test_dataloader, get_test_dataset
from kale.loaddata.molecular_datasets import graph_collate_func
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--ckpt_resume", default=None, help="path to train checkpoint file", type=str)
    parser.add_argument(
        "--save_attention", action="store_true", help="save attention matrix. use this flag to turn it on"
    )
    args = parser.parse_args()
    return args


def main():
    # ---- ignore warnings ----
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    SEED = cfg.SOLVER.SEED
    set_seed(SEED)
    pl.seed_everything(SEED, workers=True)

    # ---- setup dataset ----
    dataFolder = os.path.join(f"./datasets/" + cfg.DATA.DATASET)
    # dataFolder = os.path.join(f"./datasets/" + cfg.DATA.DATASET + "/" + str(cfg.DATA.SPLIT))
    if not os.path.exists(dataFolder):
        raise FileNotFoundError(
            f"Dataset folder {dataFolder} does not exist. Please check if the data folder exists.\n"
            f"If you haven't downloaded the data, please follow the dataset guidance at https://github.com/pykale/pykale/tree/main/examples/bindingdb_drugban#datasets"
        )
    test_dataset = get_test_dataset(dataFolder, cfg.DATA.SPLIT)

    # ---- setup dataloader ----
    test_dataloader = get_test_dataloader(
        test_dataset, batchsize=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.SOLVER.NUM_WORKERS, collate_fn=graph_collate_func
    )

    # ---- setup model ----
    if args.save_attention:
        print("Attention saving is ENABLED.")
    else:
        print("Attention saving is disabled.")
    model = get_model_from_ckpt(args.ckpt_resume, cfg, args.save_attention)
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True,
    )
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
DrugBAN on BindingDB/BioSNAP/Human drug-protein interaction prediction

Reference: https://github.com/peizhenbai/DrugBAN/blob/main/main.py
"""

import argparse
import os
import sys
import warnings
from time import time

import pandas as pd
import torch
from comet_ml.config import experiment
from configs import get_cfg_defaults
from torch.utils.data import DataLoader

from kale.embed.ban import DrugBAN
from kale.loaddata.molecular_datasets import DTIDataset, graph_collate_func
from kale.loaddata.sampler import MultiDataLoader
from kale.pipeline.drugban_trainer import Trainer
from kale.utils.seed import set_seed
from kale.utils.setup import setup_comet, setup_device


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args


def main():
    # ---- ignore warnings ----
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- setup device, configs and seed----
    device = setup_device()

    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    set_seed(cfg.SOLVER.SEED)

    # ---- setup output directory ----
    path = cfg.RESULT.OUTPUT_DIR
    path = path.strip().rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

    # ---- setup dataset ----
    dataFolder = os.path.join(f"./datasets/{cfg.DATA.DATASET}", str(cfg.DATA.SPLIT))
    names = ["train", "val", "test"] if not cfg.DA.TASK else ["source_train", "target_train", "target_test"]
    datasets = [DTIDataset(df.index, df) for df in [pd.read_csv(os.path.join(dataFolder, f"{n}.csv")) for n in names]]
    if not cfg.DA.TASK:
        train_dataset, valid_dataset, test_dataset = datasets
    else:
        train_dataset, train_target_dataset, test_target_dataset = datasets

    # ---- setup comet ----
    experiment = None
    if cfg.COMET.USE:
        hyper_params = {
            "LR": cfg.SOLVER.LEARNING_RATE,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LEARNING_RATE,
            }
            hyper_params.update(da_hyper_params)

        suffix = str(int(time() * 1000))[6:]
        experiment = setup_comet(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            log_params=hyper_params,
            experiment_tag=cfg.COMET.TAG,
            experiment_name=f"{cfg.DATA.DATASET}_{suffix}",
        )

    # ---- setup dataloader ----
    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": graph_collate_func,
    }

    if not cfg.DA.USE:  # If domain adaptation is not used
        training_generator = DataLoader(train_dataset, **params)
        params.update(shuffle=False, drop_last=False)
        data = (valid_dataset, test_dataset) if not cfg.DA.TASK else (test_target_dataset, test_target_dataset)
        valid_generator, test_generator = [DataLoader(d, **params) for d in data]
    else:  # If domain adaptation is used, and cross-domain splitting strategy is used
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        multi_generator = MultiDataLoader(
            dataloaders=[source_generator, target_generator],
            n_batches=max(len(source_generator), len(target_generator)),
        )
        params.update(shuffle=False, drop_last=False)
        valid_generator = test_generator = DataLoader(test_target_dataset, **params)

    # ---- setup model and optimizer----
    model = DrugBAN(**cfg).to(device)

    torch.backends.cudnn.benchmark = True

    # ---- setup trainer ----
    if not cfg.DA.USE:  # If domain adaptation is not used
        trainer = Trainer(
            model,
            device,
            training_generator,
            valid_generator,
            test_generator,
            experiment=experiment,
            **cfg,
        )
    else:  # If domain adaptation is used
        trainer = Trainer(
            model,
            device,
            multi_generator,
            valid_generator,
            test_generator,
            experiment=experiment,
            **cfg,
        )

    # ---- train model ----
    result = trainer.train()

    # ---- save model architecture and result ----
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == "__main__":
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

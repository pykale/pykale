import argparse
import os
import sys
import warnings
from time import time

import comet_ml
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/jiang/PycharmProjects/pykale/")
from datetime import datetime

import pytorch_lightning as pl
from configs import get_cfg_defaults
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from kale.embed.ban import DrugBAN
from kale.loaddata.drugban_datasets import DTIDataset, graph_collate_func, MultiDataLoader
from kale.pipeline.drugban_trainer_lightning import DrugbanDATrainer, DrugbanTrainer
from kale.predict.class_domain_nets import Discriminator
from kale.utils.seed import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--data", required=True, type=str, metavar="TASK", help="dataset")
    parser.add_argument(
        "--split", default="random", type=str, metavar="S", help="split task", choices=["random", "cold", "cluster"]
    )
    args = parser.parse_args()
    return args


def main():
    """The main for this DrugBAN example, showing the workflow"""
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- get args ----
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]

    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    # ---- setup dataset ----
    dataFolder = f"./datasets/{args.data}"
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        """
        'cfg.DA.TASK = False' refers to 'in-domain' splitting strategy, where
        each experimental dataset is randomly divided into training, validation,
        and test sets with a 7:1:2 ratio.
        """
        train_path = os.path.join(dataFolder, "train.csv")
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)

    else:
        """
        'cfg.DA.TASK = True' refers to 'cross-domain' splitting strategy, where
        we used the single-linkage algorithm to cluster drugs and proteins, and randomly
        selected 60% of the drug clusters and 60% of the protein clusters.

        All drug-protein pairs in the selected clusters are source domain data. The remaining drug-protein pairs are target domain data.

        In the setting of domain adaptation, all labelled source domain data and 80% unlabelled target domain data are used for training.
        the remaining 20% labelled target domain data are used for testing.
        """
        train_source_path = os.path.join(dataFolder, "source_train.csv")
        train_target_path = os.path.join(dataFolder, "target_train.csv")
        test_target_path = os.path.join(dataFolder, "target_test.csv")
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    # ---- setup dataloader params ----
    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": graph_collate_func,
    }

    # ---- setup dataloader ----
    if not cfg.DA.USE:
        """If domain adaptation is not used"""
        training_generator = DataLoader(train_dataset, **params)
        params["shuffle"] = False
        params["drop_last"] = False
        if not cfg.DA.TASK:
            """If in-domain splitting strategy is used"""
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            """If cross-domain splitting strategy is used"""
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        """If domain adaptation is used, and cross-domain splitting strategy is used"""
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)

        params["shuffle"] = False
        params["drop_last"] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    # ---- set logger ----
    experiment_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if cfg.COMET.USE:
        logger = CometLogger(
            api_key=cfg["COMET"]["API_KEY"],
            project_name=cfg["COMET"]["PROJECT_NAME"],
            experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, experiment_time),
        )
    else:
        logger = TensorBoardLogger(save_dir=cfg.RESULT.OUTPUT_DIR, name=experiment_time)

    # ---- setup model ----
    if not cfg.DA.USE:
        model = DrugbanTrainer(model=DrugBAN(**cfg), **cfg)
        trainer = pl.Trainer(
            devices="auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=cfg["SOLVER"]["MAX_EPOCH"],
            logger=logger,
        )
        trainer.fit(model, train_dataloaders=training_generator, val_dataloaders=val_generator)
        trainer.test(model, dataloaders=test_generator)
    else:
        model = DrugbanDATrainer(model=DrugBAN(**cfg), discriminator=Discriminator, **cfg)
        trainer = pl.Trainer(
            devices="auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=cfg["SOLVER"]["MAX_EPOCH"],
            logger=logger,
        )
        trainer.fit(model, train_dataloaders=multi_generator, val_dataloaders=val_generator)
        trainer.test(model, dataloaders=test_generator)


if __name__ == "__main__":
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

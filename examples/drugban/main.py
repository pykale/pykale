import argparse
import os
import sys
import warnings
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/jiang/Documents/repositories/pykale/")

from configs import get_cfg_defaults

from kale.embed.drugban import DrugBAN
from kale.loaddata.drugban_datasets import DTIDataset, MultiDataLoader
from kale.pipeline.drugban_trainer import Trainer
from kale.predict.class_domain_nets import Discriminator
from kale.utils.drugban_utils import graph_collate_func, mkdir
from kale.utils.seed import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument("--cfg", required=True, help="path to config file", type=str)
parser.add_argument("--data", required=True, type=str, metavar="TASK", help="dataset")
parser.add_argument(
    "--split", default="random", type=str, metavar="S", help="split task", choices=["random", "cold", "cluster"]
)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)

    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f"./datasets/{args.data}"
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
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
        train_source_path = os.path.join(dataFolder, "source_train.csv")
        train_target_path = os.path.join(dataFolder, "target_train.csv")
        test_target_path = os.path.join(dataFolder, "target_test.csv")
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
        train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
        test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False,
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
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
                "DA_optim_lr": cfg.SOLVER.DA_LR,
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": graph_collate_func,
    }

    if not cfg.DA.USE:
        training_generator = DataLoader(train_dataset, **params)
        params["shuffle"] = False
        params["drop_last"] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
        params["shuffle"] = False
        params["drop_last"] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(
                input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"], n_class=cfg["DECODER"]["BINARY"]
            ).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(
            model,
            opt,
            device,
            training_generator,
            val_generator,
            test_generator,
            opt_da=None,
            discriminator=None,
            experiment=experiment,
            **cfg,
        )
    else:
        trainer = Trainer(
            model,
            opt,
            device,
            multi_generator,
            val_generator,
            test_generator,
            opt_da=opt_da,
            discriminator=domain_dmm,
            experiment=experiment,
            **cfg,
        )
    result = trainer.train()

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

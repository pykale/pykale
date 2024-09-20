import argparse
import os
import sys
import warnings
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/jiang/Documents/repositories/pykale/")

from comet_ml import Experiment
from configs import get_cfg_defaults

from kale.embed.ban import DrugBAN
from kale.loaddata.drugban_datasets import DTIDataset, graph_collate_func, MultiDataLoader
from kale.pipeline.drugban_trainer import Trainer
from kale.predict.class_domain_nets import Discriminator
from kale.utils.seed import set_seed


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

    # ---- setup device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ---- ignore warnings ----
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- get args ----
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]

    # ---- setup output directory ----
    path = cfg.RESULT.OUTPUT_DIR
    path = path.strip().rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

    # ---- setup comet ----
    experiment = None
    comet_support = True

    # ---- print information ----
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

        All drug-protein pairs in the selected clusters are source domain data.
        The remaining drug-protein pairs are target domain data.

        In the setting of domain adaptation, all labelled source domain data and 80% unlabelled
        target domain data are used for training. The remaining 20% labelled target domain data are used for testing.
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

    # ---- setup comet ----
    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
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
        # If domain adaptation is not used
        training_generator = DataLoader(train_dataset, **params)
        params["shuffle"] = False
        params["drop_last"] = False
        if not cfg.DA.TASK:
            # If in-domain splitting strategy is used
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            # If cross-domain splitting strategy is used
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)
    else:
        # If domain adaptation is used, and cross-domain splitting strategy is used
        source_generator = DataLoader(train_dataset, **params)
        target_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_generator), len(target_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)

        params["shuffle"] = False
        params["drop_last"] = False
        val_generator = DataLoader(test_target_dataset, **params)
        test_generator = DataLoader(test_target_dataset, **params)

    # ---- setup model and optimizer----
    model = DrugBAN(**cfg).to(device)

    if cfg.DA.USE:
        # If domain adaptation is used
        if cfg["DA"]["RANDOM_LAYER"]:
            # Initialize the Discriminator with an input size from the random dimension specified in the config
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            # Initialize the Discriminator with an input size derived from the decoder's input dimension
            domain_dmm = Discriminator(
                input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"], n_class=cfg["DECODER"]["BINARY"]
            ).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())

        # Initialize the optimizer for the DrugBAN model
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        # Initialize the optimizer for the Domain Discriminator model
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        # If domain adaptation is not used, only initialize the optimizer for the DrugBAN model
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    # ---- setup trainer ----
    if not cfg.DA.USE:
        # If domain adaptation is not used
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
        # If domain adaptation is used
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

import argparse
import warnings

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import get_model, get_supervertex
from torch.utils.data import DataLoader

import kale.utils.seed as seed
from kale.loaddata.polypharmacy_datasets import PolypharmacyDataset
from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex

warnings.filterwarnings(action="ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description="GripNet Training for Polypharmacy Side Effect Prediction")
    parser.add_argument("--cfg", type=str, default="config.yaml", help="config file path")
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed.set_seed(cfg.SOLVER.SEED)

    # ---- setup dataset and data loader ----
    train_dataset = PolypharmacyDataset(cfg.DATASET, mode="train")
    dataloader_train = DataLoader(train_dataset, batch_size=1)

    # ---- setup supergraph ----
    # create protein and drug supervertex
    supervertex_protein = SuperVertex("protein", train_dataset.protein_feat, train_dataset.protein_edge_index)
    supervertex_drug = SuperVertex("drug", train_dataset.drug_feat, train_dataset.edge_index, train_dataset.edge_type)

    # create superedge form protein to drug supervertex
    superedge = SuperEdge("protein", "drug", train_dataset.protein_drug_edge_index)

    setting_protein = get_supervertex(cfg.GRIPN_SV1)
    setting_drug = get_supervertex(cfg.GRIPN_SV2)

    # construct supergraph
    supergraph = SuperGraph([supervertex_protein, supervertex_drug], [superedge])
    supergraph.set_supergraph_para_setting([setting_protein, setting_drug])

    # ---- setup model and trainer ----
    model = get_model(supergraph, cfg)
    print(model)

    trainer = pl.Trainer(
        default_root_dir=cfg.OUTPUT_DIR,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        log_every_n_steps=cfg.SOLVER.LOG_EVERY_N_STEPS,
    )

    # ---- train, validate and test ----
    # The training set is reused here as the validation and test sets for usage demonstration. See ReadMe for details.
    trainer.fit(model, dataloader_train, dataloader_train)
    _ = trainer.test(model, dataloader_train)


if __name__ == "__main__":
    main()

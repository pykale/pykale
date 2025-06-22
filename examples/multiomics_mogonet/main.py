"""
Multiomics Integration via Graph Convolutional Networks for Cancer Classification Tasks.

Reference:
Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021). MOGONET integrates multi-omics data
using graph convolutional networks allowing patient classification and biomarker identification. Nature communications.
https://www.nature.com/articles/s41467-021-23774-w
"""

import argparse
import logging
import warnings

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import MogonetModel

import kale.utils.seed as seed
from kale.interpret.model_weights import select_top_features
from kale.loaddata.multiomics_datasets import SparseMultiomicsDataset
from kale.prepdata.tabular_transform import ToOneHotEncoding, ToTensor

warnings.filterwarnings(action="ignore")


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="MOGONET Training for Multiomics Data Integration")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()

    return args


def main():
    args = arg_parse()

    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n==> Using device " + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    seed.set_seed(cfg.SOLVER.SEED)

    # ---- setup dataset ----
    print("\n==> Preparing dataset...")
    file_names = []
    for modality in range(1, cfg.DATASET.NUM_MODALITIES + 1):
        file_names.append(f"{modality}_tr.csv")
        file_names.append(f"{modality}_lbl_tr.csv")
        file_names.append(f"{modality}_te.csv")
        file_names.append(f"{modality}_lbl_te.csv")
        file_names.append(f"{modality}_feat_name.csv")

    multiomics_data = SparseMultiomicsDataset(
        root=cfg.DATASET.ROOT,
        raw_file_names=file_names,
        num_modalities=cfg.DATASET.NUM_MODALITIES,
        num_classes=cfg.DATASET.NUM_CLASSES,
        edge_per_node=cfg.MODEL.EDGE_PER_NODE,
        url=cfg.DATASET.URL,
        random_split=cfg.DATASET.RANDOM_SPLIT,
        equal_weight=cfg.MODEL.EQUAL_WEIGHT,
        pre_transform=ToTensor(dtype=torch.float),
        target_pre_transform=ToOneHotEncoding(dtype=torch.float),
    )

    print(multiomics_data)

    # ---- setup model ----
    print("\n==> Building model...")
    mogonet_model = MogonetModel(cfg, dataset=multiomics_data)
    print(mogonet_model)

    # ---- setup pretrain model and trainer ----
    print("\n==> Pretrain GCNs...")
    model = mogonet_model.get_model(pretrain=True)
    trainer_pretrain = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS_PRETRAIN,
        default_root_dir=cfg.OUTPUT.OUT_DIR,
        accelerator="auto",
        devices="auto",
        enable_model_summary=False,
    )
    trainer_pretrain.fit(model)

    # ---- set train model and trainer ----
    print("\n==> Training model...")
    model = mogonet_model.get_model(pretrain=False)
    trainer = pl.Trainer(
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT.OUT_DIR,
        accelerator="auto",
        devices="auto",
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    # ---- testing model ----
    print("\n==> Testing model...")
    _ = trainer.test(model)

    print("\n==> Identifying biomarkers...")
    pl_logger = logging.getLogger("pytorch_lightning")
    pl_logger.setLevel(logging.ERROR)
    trainer.progress_bar_callback.disable()
    f1_key = "F1" if multiomics_data.num_classes == 2 else "F1 macro"
    df_featimp_top = select_top_features(
        trainer=trainer,
        model=model,
        dataset=multiomics_data,
        metric=f1_key,
        num_top_feats=30,
        verbose=False,
    )

    print("{:>4}\t{:<20}\t{:>5}\t{}".format("Rank", "Feature name", "Omics", "Importance"))
    for rank, row in enumerate(df_featimp_top.itertuples(index=False), 1):
        print(f"{rank:>4}\t{row.feat_name:<20}\t{row.omics:>5}\t{row.imp:.4f}")


if __name__ == "__main__":
    main()

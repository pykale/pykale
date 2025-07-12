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
from tqdm import tqdm

from examples.bindingdb_drugban.model import get_model_from_ckpt, get_test_dataloader, get_test_dataset
from kale.loaddata.molecular_datasets import graph_collate_func
from kale.utils.seed import set_seed


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument("--ckpt_resume", default=None, help="path to train checkpoint file", type=str)
    parser.add_argument("--save_att_path", default=None, help="path to save attention maps", type=str)
    args = parser.parse_args()
    return args


def main():
    # ---- ignore warnings ----
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    # ---- setup configs ----
    args = arg_parse()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.BATCH_SIZE = 1

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
    model = get_model_from_ckpt(args.ckpt_resume, cfg)
    model.model.eval()

    # ---- test the model ----
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True,
    )
    trainer.test(model, dataloaders=test_dataloader)

    # ---- save attention maps ----
    print("Attention saving is ENABLED.") if args.save_att_path is not None else print("Attention saving is DISABLED.")
    if args.save_att_path is not None:
        all_attentions = []
        for batch in tqdm(test_dataloader):
            drug, protein, _ = batch
            drug, protein = drug.to(model.device), protein.to(model.device)

            _, _, _, _, attention = model.model.forward(drug, protein, mode="eval")  # [B, H, V, Q]

            attention = attention.detach().cpu()
            all_attentions.append(attention)

        # Concatenate into one tensor: [N, H, V, Q]
        all_attentions = torch.cat(all_attentions, dim=0)
        torch.save(all_attentions, args.save_att_path)


if __name__ == "__main__":
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

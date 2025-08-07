import argparse
import os
import json
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
import random

# from loaddata.dataloader import get_train_val_test_loader
from pipeline.cgcnn.model_cgcnn import get_cgcnn_model  # Ensure the correct module and function path
from loaddata.cifdata import CIFData
from loaddata.collate import collate_pool_leftnet
from pipeline.leftnet.model_leftnet import get_leftnet_model
from config import get_cfg_defaults
# from tests.shap_utils import compute_shap_values  # Import the SHAP utility function
from pipeline.cartnet.model_cartnet import get_cartnet_model  # Import the missing function



def arg_parse():
    parser = argparse.ArgumentParser(description="Test a model from a checkpoint.")
    parser.add_argument("--cfg", required=True, type=str, help="Path to the config file used to train the model.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the checkpoint file (.ckpt).")
    parser.add_argument("--cif_folder", required=True, type=str, help="Path to the folder containing CIF files.")
    parser.add_argument("--test_data", required=True, type=str, help="Path to the JSON file containing test data.")
    parser.add_argument("--devices",
                        default=1,
                        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU. str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs")
    parser.add_argument("--output_dir", default="output_test", type=str, help="Directory to save test results.")
    parser.add_argument("--output_file", default="test_predictions.csv", type=str, help="Name of the output CSV file.")
    return parser.parse_args()


def load_model_and_data(cfg, checkpoint_path, test_data_path, cif_folder):
    """Load model and test data according to the provided configuration and checkpoint."""
    # Set random seed for reproducibility
    seed = cfg.SOLVER.SEED if hasattr(cfg.SOLVER, 'SEED') else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    test_data = pd.DataFrame.from_dict(test_data, orient='index')
    test_data.reset_index(inplace=True)
    test_data.rename(columns={'index': 'mpids'}, inplace=True)

    # Prepare dataset
    test_data = test_data[['mpids', 'bg']]  # assuming 'bg' is your target property
    test_dataset = CIFData(
        test_data,
        # cfg.MODEL.CIF_FOLDER,
        cif_folder,
        cfg.MODEL.INIT_FILE,
        cfg.MODEL.MAX_NBRS,
        cfg.MODEL.RADIUS,
        cfg.SOLVER.RANDOMIZE
    )

    # Get the structure information from one example to complete config
    structures = test_dataset[0]
    orig_atom_fea_len = structures.atom_fea.shape[-1]
    nbr_fea_len = structures.nbr_fea.shape[-1]
    pos_fea_len = structures.positions.shape[-1]
    cfg.defrost()
    cfg.GRAPH.ORIG_ATOM_FEA_LEN = orig_atom_fea_len
    cfg.GRAPH.NBR_FEA_LEN = nbr_fea_len
    cfg.GRAPH.POS_FEA_LEN = pos_fea_len
    cfg.freeze()

    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_pool_leftnet,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.SOLVER.WORKERS,
    )

    # Initialize model
    if cfg.MODEL.NAME == "cgcnn":
        model = get_cgcnn_model(cfg)
    elif cfg.MODEL.NAME == "leftnet":
        model = get_leftnet_model(cfg)
    elif cfg.MODEL.NAME == "cartnet":
        model = get_cartnet_model(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

    # Load the checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model, test_loader, test_data


def main():
    args = arg_parse()

    # Load and modify config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Prepare model and data
    model, test_loader, test_data = load_model_and_data(cfg, args.checkpoint, args.test_data, args.cif_folder)

    # Setup trainer for testing
    trainer = pl.Trainer(
        accelerator="gpu" if int(args.devices) > 0 else "cpu",
        devices=args.devices if int(args.devices) > 0 else None
    )

    # Test the model
    test_results = trainer.test(model, dataloaders=test_loader)
    print("Test Results:", test_results)

    # Compute SHAP values
    # compute_shap_values(model, test_loader, is_gnn=(cfg.MODEL.NAME in ["cgcnn", "leftnet"]))

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            preds, *_ = model(batch)  # grab only the first value
            preds = preds.cpu().numpy()

            all_predictions.extend(preds.flatten())
    test_data["prediction"] = all_predictions
    output_file_path = os.path.join(args.output_dir, args.output_file)
    test_data.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    main()

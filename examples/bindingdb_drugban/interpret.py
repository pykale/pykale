import argparse
import os

import pandas as pd
import torch
from rdkit import Chem

from kale.interpret.visualize import draw_attention_map, draw_mol_with_attention
from kale.prepdata.tensor_reshape import normalize_tensor


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Interpretation and visualization for DrugBAN")
    parser.add_argument("--att_file", required=True, default="att_map.pt", help="path to attention files", type=str)
    parser.add_argument("--data_file", required=True, default=None, help="path to smiles file", type=str)
    parser.add_argument("--out_path", default="./visualization", help="path to save visualization", type=str)

    args = parser.parse_args()
    return args


def get_real_length(smile, protein_sequence):
    """Get the real length of the drug and protein sequences."""
    mol = Chem.MolFromSmiles(smile)
    return mol.GetNumAtoms(), len(protein_sequence)


def process_sample(index, attention, smile, protein, out_dir):
    """Process a single sample to visualize attention and molecule."""
    att = attention[index]  # [H, V, Q]
    real_drug_len, real_prot_len = get_real_length(smile, protein)
    att = att[:, :real_drug_len, :real_prot_len].mean(0)  # [V, Q]

    # Normalize
    att = normalize_tensor(att)

    # Save plots
    os.makedirs(out_dir, exist_ok=True)
    att_path = os.path.join(out_dir, f"att_map_{index}.png")
    mol_path = os.path.join(out_dir, f"mol_{index}.svg")

    draw_attention_map(att, att_path, title=f"Drug {index} Attention", xlabel="Drug Tokens", ylabel="Protein Tokens")
    draw_mol_with_attention(att.mean(dim=1), smile, mol_path)


def main():
    args = arg_parse()
    attention = torch.load(args.att_file, map_location="cpu")  # [B, H, V, Q]
    data_df = pd.read_csv(args.data_file)
    smiles = data_df["SMILES"]
    proteins = data_df["Protein"]

    for i in range(len(attention)):
        process_sample(i, attention, smiles[i], proteins[i], args.out_path)


if __name__ == "__main__":
    main()

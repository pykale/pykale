import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import colormaps
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def normalize_tensor(tensor, eps=1e-8):
    """Normalize a tensor to [0, 1] range."""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def draw_attention_map(attention_weights, save_path, title="Mean Attention Map"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.numpy(), cmap="viridis")
    plt.xlabel("Protein sequence tokens")
    plt.ylabel("Drug tokens")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def draw_mol_with_attention(attention_weights, smile, out_path):
    mol = Chem.MolFromSmiles(smile)
    weights = normalize_tensor(attention_weights)
    weights = weights.cpu().numpy().tolist()

    # Draw with RDKit 2D drawer
    cmap = colormaps["viridis"]
    atom_colors = {i: cmap(float(w))[:3] for i, w in enumerate(weights)}

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii={i: 0.3 for i in atom_colors},
    )
    drawer.FinishDrawing()

    with open(out_path, "w") as f:
        f.write(drawer.GetDrawingText())


def get_real_length(smile, protein_sequence):
    mol = Chem.MolFromSmiles(smile)
    return mol.GetNumAtoms(), len(protein_sequence)


def process_sample(index, attention, smile, protein, out_dir):
    att = attention[index]  # [H, V, Q]
    real_drug_len, real_prot_len = get_real_length(smile, protein)
    att = att[:, :real_drug_len, :real_prot_len].mean(0)  # [V, Q]

    # Normalize
    att = normalize_tensor(att)

    # Save plots
    os.makedirs(out_dir, exist_ok=True)
    att_path = os.path.join(out_dir, f"att_map_{index}.png")
    mol_path = os.path.join(out_dir, f"mol_{index}.svg")

    draw_attention_map(att, att_path, title=f"Drug {index} Attention")
    draw_mol_with_attention(att.mean(dim=1), smile, mol_path)


def main():
    attention_path = "attention_map.pt"  # Path to the attention map file
    data_path = "datasets/bindingdb/target_test.csv"  # Path to the SMILES file
    out_dir = "./visualization"
    max_n = 5

    attention = torch.load(attention_path, map_location="cpu")  # [B, H, V, Q]
    data_df = pd.read_csv(data_path)
    smiles = data_df["SMILES"]
    proteins = data_df["Protein"]

    for i in range(min(max_n, len(attention))):
        process_sample(i, attention, smiles[i], proteins[i], out_dir)


if __name__ == "__main__":
    main()

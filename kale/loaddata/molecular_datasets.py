# =============================================================================
# Author: Jiayang Zhang, jiayang.zhang@sheffield.ac.uk
# =============================================================================

"""
Dataset setting and data loader for BindingDB, BioSNAP and Human datasets,
by refactoring  https://github.com/peizhenbai/DrugBAN/blob/main/dataloader.py
"""


import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops

from kale.prepdata.chem_transform import integer_label_protein


def graph_collate_func(x):
    d, p, y = zip(*x)
    d = Batch.from_data_list(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def smiles_to_graph(smiles, max_drug_nodes):
    mol = Chem.MolFromSmiles(smiles)

    atom_features = []  # shape: (num_atoms, num_atom_features)
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),  # Atomic number - essential
                atom.GetDegree(),
                atom.GetImplicitValence(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
            ]
        )
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []  # shape: (2, num_edges) --> each is [source, target]
    edge_features = []  # shape: (num_edges, num_edge_features)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[start, end], [end, start]]
        edge_features += [
            [bond.GetBondTypeAsDouble()],
            [bond.GetBondTypeAsDouble()],
        ]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    # add self-loops
    edge_index, edge_features = add_self_loops(
        edge_index=edge_index, edge_attr=edge_features, num_nodes=atom_features.size(0), fill_value=0.0
    )

    # Make sure edge_attr is always 2D
    if edge_features.ndim == 1:
        edge_features = edge_features.unsqueeze(-1)

    # Note: one-hot encoding is used for atom and bond features in dgl, while pyG uses float

    # ======== Padding: Add virtual nodes ===========
    num_actual_nodes = atom_features.size(0)
    if num_actual_nodes < max_drug_nodes:
        num_virtual_nodes = max_drug_nodes - num_actual_nodes

        virtual_node_features = torch.zeros(num_virtual_nodes, atom_features.size(1))
        atom_features = torch.cat([atom_features, virtual_node_features], dim=0)
        # Note: No new edges are added for virtual nodes

    assert atom_features.ndim == 2, f"x must be 2D, got shape {atom_features.shape}"
    assert edge_features.ndim == 2, f"edge_attr must be 2D, got shape {edge_features.shape}"

    graph = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=max_drug_nodes)

    return graph


class DTIDataset(Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        """
        Initializes the DTIDataset.

        Parameters:
        -----------
        list_IDs : list
            List of indices corresponding to the rows in the DataFrame `df` that will be used by the dataset.

        df : pandas.DataFrame
            The DataFrame containing the drug-target interaction data.

        max_drug_nodes : int, optional
            Maximum number of nodes for the molecular graphs. Default is 290.
        """
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
        int
            The total number of samples.
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Returns a single drug-target pair sample.

        Parameters:
        -----------
        index : int
            The index of the sample to retrieve.

        Returns:
        --------
        v_drug : DGLGraph
            A tensor representing the drug molecule, with node features and optional virtual nodes.

        v_protein : torch.Tensor
            A tensor representing the encoded protein sequence.

        y : float
            The label for this drug-protein pair.
        """

        # Retrieve the actual index in the DataFrame from the list of IDs
        index = self.list_IDs[index]

        # Get SMILES
        v_drug = self.df.iloc[index]["SMILES"]
        v_drug = smiles_to_graph(smiles=v_drug, max_drug_nodes=self.max_drug_nodes)

        # Extract the protein sequence and convert it to a tensor of integers
        v_protein = self.df.iloc[index]["Protein"]
        v_protein = integer_label_protein(v_protein)

        # Extract the label
        y = self.df.iloc[index]["Y"]

        return v_drug, v_protein, y

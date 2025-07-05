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
    """
    Custom collate function for PyTorch DataLoader to batch drug-protein interaction samples.

    Each sample in the input list `x` is a tuple containing:
        - a PyTorch Geometric `Data` object representing a drug molecular graph,
        - a protein sequence represented as a tensor or array,
        - a label (e.g., interaction score or binary classification target).

    This function:
        - batches the molecular graphs using `Batch.from_data_list`,
        - stacks the protein tensors into a single tensor,
        - stacks the labels into a single tensor.

    Parameters:
    -----------
    x : list of tuples
        Each tuple contains (drug_graph, protein_tensor, label).

    Returns:
    --------
    drug : torch_geometric.data.Batch
        A batched PyTorch Geometric Batch object of drug molecular graphs.

    protein : torch.Tensor
        A 2D tensor of protein sequence features, shape (batch_size, sequence_length).

    label : torch.Tensor
        A 1D or 2D tensor of labels, depending on the task.
    """
    drug, protein, label = zip(*x)
    drug = Batch.from_data_list(drug)
    return drug, torch.tensor(np.array(protein)), torch.tensor(label)


def smiles_to_graph(smiles, max_drug_nodes):
    """
    Converts a SMILES string into a padded PyTorch Geometric molecular graph.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.
    max_drug_nodes : int
        Maximum number of nodes in the graph. If the actual number is smaller, virtual (zero-feature) nodes are added.

    Returns
    -------
    Data
        A PyTorch Geometric `Data` object containing:
        - x: Node feature matrix
        - edge_index: Edge connectivity
        - edge_attr: Edge feature matrix
        - num_nodes: Total number of nodes (including virtual nodes)
    """
    mol = Chem.MolFromSmiles(smiles)

    atom_features = []  # shape: (num_atoms, num_atom_features)
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),  # Atomic number - essential
                atom.GetDegree(),
                atom.GetValence(getExplicit=False),
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
    def __init__(self, list_ids, df, max_drug_nodes=290):
        """
        Initializes the DTIDataset.

        Parameters:
        -----------
        list_ids : list
            List of indices corresponding to the rows in the DataFrame `df` that will be used by the dataset.

        df : pandas.DataFrame
            The DataFrame containing the drug-target interaction data.

        max_drug_nodes : int, optional
            Maximum number of nodes for the molecular graphs. Default is 290.
        """
        self.list_ids = list_ids
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
        return len(self.list_ids)

    def __getitem__(self, index):
        """
        Returns a single drug-target pair sample.

        Parameters:
        -----------
        index : int
            The index of the sample to retrieve.

        Returns:
        --------
        vec_drug : torch_geometric.data.Data
            A PyG Data object representing the drug molecule, with node features and optional virtual nodes.

        vec_protein : torch.Tensor
            A tensor representing the encoded protein sequence.

        y : float
            The label for this drug-protein pair.
        """

        # Retrieve the actual index in the DataFrame from the list of ids
        index = self.list_ids[index]

        # Get SMILES
        vec_drug = self.df.iloc[index]["SMILES"]
        vec_drug = smiles_to_graph(smiles=vec_drug, max_drug_nodes=self.max_drug_nodes)

        # Extract the protein sequence and convert it to a tensor of integers
        vec_protein = self.df.iloc[index]["Protein"]
        vec_protein = integer_label_protein(vec_protein)

        # Extract the label
        y = self.df.iloc[index]["Y"]

        return vec_drug, vec_protein, y

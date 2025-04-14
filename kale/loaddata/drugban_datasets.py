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
    Collate function for batching a list of samples (graph, protein, label).

    Parameters
    ----------
    x : list of tuples
        Each tuple contains (graph, protein_tensor, label).

    Returns
    -------
    tuple
        A tuple of:
        - Batched graph as torch_geometric.data.Batch
        - Tensor of protein sequences (batch_size, sequence_length)
        - Tensor of labels (batch_size,)
    """
    d, p, y = zip(*x)
    d = Batch.from_data_list(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def smiles_to_graph(smiles, max_drug_nodes):
    """
    Converts a SMILES string into a PyG graph with atom and bond features,
    and pads the graph to a fixed number of nodes.

    Parameters
    ----------
    smiles : str
        SMILES string representing a drug molecule.

    max_drug_nodes : int
        Maximum number of nodes for graph padding.

    Returns
    -------
    graph : torch_geometric.data.Data
        A graph object with atom features `x`, edge index `edge_index`,
        edge attributes `edge_attr`, and fixed `num_nodes`.
    """
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
        v_d : torch_geometric.data.Data

        v_p : torch.Tensor
            A tensor representing the encoded protein sequence.

        y : float
            The label for this drug-protein pair.
        """

        # Retrieve the actual index in the DataFrame from the list of IDs
        index = self.list_IDs[index]

        # Get SMILES
        v_d = self.df.iloc[index]["SMILES"]
        v_d = smiles_to_graph(smiles=v_d, max_drug_nodes=self.max_drug_nodes)

        # Extract the protein sequence and convert it to a tensor of integers
        v_p = self.df.iloc[index]["Protein"]
        v_p = integer_label_protein(v_p)

        # Extract the label
        y = self.df.iloc[index]["Y"]

        return v_d, v_p, y


# train_path = os.path.join("/home/jiang/PycharmProjects/pykale/examples/bindingdb_drugban/datasets/biosnap/random", "train.csv")
# df_train = pd.read_csv(train_path)
# train_dataset = DTIDataset(df_train.index.values, df_train).__getitem__(index=1)


class MultiDataLoader(object):
    """
     A class to iterate over multiple DataLoader objects in parallel.


    Args:
        _dataloaders (list): A list of DataLoader objects.
        _n_batches (int): The number of batches to iterate over.
        _iterators (list): A list of iterators corresponding to the DataLoaders.

    """

    def __init__(self, dataloaders, n_batches):
        """Initialise the MultiDataLoader

        Args:
            dataloaders (list): A list of DataLoader objects to iterate over.
            n_batches (int): The number of batches to iterate. Must be greater than 0.

        Raises:
            ValueError: If n_batches is less than or equal to 0.
        """
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        """Initializes iterators for each DataLoader."""
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        """Get the next batch from each DataLoader."""

        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        """Iterates over the DataLoader objects

        Yields:
            list: A list of batches, one from each DataLoader, in each iteration.
        """
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        """Returns the number of batches"""
        return self._n_batches

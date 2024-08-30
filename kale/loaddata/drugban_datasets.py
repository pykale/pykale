from functools import partial

import numpy as np
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from torch.utils.data import Dataset

from kale.utils.drugban_utils import integer_label_protein
import dgl
import dgl

def graph_collate_func(x):
    """
    For torch.utils.data.DataLoader, collates a list of graph data into a batch.

    Args:
        x (list): A list of tuples, where each tuple contains:
            - d (dgl.DGLGraph): A drug representation as a DGL graph object.
            - p (np.ndarray or list): A protein representation.
            - y (int or float): A target label.

    Returns:
        tuple:
            - dgl.DGLGraph: A batched DGL graph of the drug representations.
            - torch.Tensor: A tensor of batched protein representations.
            - torch.Tensor: A tensor of batched target labels.
    """
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


class DTIDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling drug-target interaction (DTI) data.

    This dataset is designed to preprocess and provide molecular graphs (from SMILES strings) and
    protein sequence information for deep learning models. It handles both drug molecules and their
    corresponding target proteins, and includes additional processing to manage varying graph sizes
    by adding virtual nodes.

    Attributes:
    -----------
    list_IDs : list
        A list of indices corresponding to the rows in the DataFrame `df` that this dataset will handle.

    df : pandas.DataFrame
        A DataFrame containing the dataset. It must include columns for 'SMILES' (for drug molecules)
        and 'Protein' (for target proteins). The 'Y' column contains the label.

    max_drug_nodes : int, optional
        The maximum number of nodes allowed for any drug graph. If a graph has fewer nodes, virtual nodes
        will be added to match this number. Defaults to 290.

    atom_featurizer : dgllife.utils.CanonicalAtomFeaturizer
        An instance of CanonicalAtomFeaturizer used to generate atom features for the drug molecules.

    bond_featurizer : dgllife.utils.CanonicalBondFeaturizer
        An instance of CanonicalBondFeaturizer used to generate bond features for the drug molecules.

    fc : partial
        A partially applied function (`smiles_to_bigraph`) that converts SMILES strings to molecular
        graphs with additional processing (like adding self-loops).
    """
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
             Maximum number of nodes for the drug molecular graphs. Default is 290.
         """
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

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
        v_d : DGLGraph
            A tensor representing the drug molecule, with node features and optional virtual nodes.

        v_p : torch.Tensor
            A tensor representing the encoded protein sequence.

        y : float
            The label for this drug-protein pair.
        """
        # Retrieve the actual index in the DataFrame from the list of IDs
        index = self.list_IDs[index]

        # Convert SMILES to a molecular graph (DGLGraph) with atom and bond features
        v_d = self.df.iloc[index]["SMILES"]
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

        # Extract node features and determine the number of actual nodes
        actual_node_feats = v_d.ndata.pop("h")
        num_actual_nodes = actual_node_feats.shape[0]

        # Calculate the number of virtual nodes required to match `max_drug_nodes`
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes

        # Create virtual nodes (of zeros) and add them to the actual node features
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        # Assign the updated node features back to the graph
        v_d.ndata["h"] = actual_node_feats # This is a tensor

        # Create features for virtual nodes
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        # Add virtual nodes to the graph
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        # Add self-loops to the graph
        v_d = v_d.add_self_loop()

        # Extract the protein sequence and convert it to a tensor of integers
        v_p = self.df.iloc[index]["Protein"]
        v_p = integer_label_protein(v_p)

        # Extract the label
        y = self.df.iloc[index]["Y"]

        return v_d, v_p, y


class MultiDataLoader(object):
    """
     A class to iterate over multiple DataLoader objects in parallel.


    Args:
        _dataloaders (list): A list of DataLoader objects.
        _n_batches (int): The number of batches to iterate over.
        _iterators (list): A list of iterators corresponding to the DataLoaders.

    """
    def __init__(self, dataloaders, n_batches):
        """ Initialise the MultiDataLoader

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

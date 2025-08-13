import json
import os
import warnings

import torch
import numpy as np
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from kale.evaluate.metrics import GaussianDistance
from kale.prepdata.materials_features import AtomCustomJSONInitializer


class CIFData(Dataset):
    """
    Dataset class for loading and processing crystal structures from CIF files.

    Each item corresponds to a single crystal graph, parsed from a CIF file and
    transformed into atom-level and neighbor-level features for use in 
    crystal graph neural networks.

    Args:
        mpids_bg (pd.DataFrame): DataFrame containing columns ['mpids', 'bg'] with IDs and target values.
        cif_folder (str): Path to directory containing .cif files.
        init_file (str): Path to JSON file defining atomic feature vectors.
        max_nbrs (int): Maximum number of neighbors per atom.
        radius (float): Cutoff radius for constructing neighbor graphs.
        randomize (bool): Whether to shuffle the dataset order.
        dmin (float): Minimum distance for Gaussian basis expansion.
        step (float): Step size for the Gaussian distance expansion.
    
    Returns:
        CIFDataItem: An object containing atom features, neighbor features, 
                     atomic positions, atomic numbers, target value, and cif ID.

    """

    def __init__(self, mpids_bg, cif_folder, init_file, max_nbrs, radius, randomize, dmin=0, step=0.2):

        self.max_num_nbr, self.radius = max_nbrs, radius

        if randomize:
            self.mpids_bg_dataset = mpids_bg.sample(frac=1).reset_index(
                drop=True).values  # Shuffling data and converting df to array
        else:
            self.mpids_bg_dataset = mpids_bg.reset_index(drop=True).values

        atom_init_file = init_file
        self.cif_folder = cif_folder
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.mpids_bg_dataset)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):

        cif_id, target_np = self.mpids_bg_dataset[idx]
        crystal = Structure.from_file(os.path.join(self.cif_folder,
                                                   cif_id + '.cif'))
        atom_fea_np = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                 for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea_np)
        target = torch.Tensor([float(target_np)])
        positions = torch.Tensor(crystal.cart_coords)
        atom_num = torch.Tensor(crystal.atomic_numbers).long()

        all_nbrs = crystal.get_all_neighbors(self.radius)  # include_index is depreciated. Index is always included now.
        all_nbrs = [sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in
                    all_nbrs]  # Sorts nbrs based on the value of key as applied to each element of the list.
        nbr_fea_idx, nbr_fea_np = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_np.append(list(map(lambda x: x[1], nbr)) +
                                  [self.radius + 1.] * (self.max_num_nbr -
                                                        len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x.index,
                                            nbr[:self.max_num_nbr])))
                nbr_fea_np.append(list(map(lambda x: x.nn_distance,
                                           nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea_np = np.array(nbr_fea_idx), np.array(nbr_fea_np)
        nbr_fea_gp = self.gdf.expand(nbr_fea_np)

        nbr_fea = torch.Tensor(nbr_fea_gp)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        # Reshape neighbor info into edge_index / edge_attr
        src_nodes = torch.arange(atom_fea.shape[0]).repeat_interleave(self.max_num_nbr)
        dst_nodes = nbr_fea_idx.view(-1)
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)    # [2, E]
        edge_attr = nbr_fea.view(-1, nbr_fea.shape[-1])      # [E, edge_feat_dim]
        max_nbrs = nbr_fea_idx.shape[-1]

        return Data(
        x=atom_fea,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=positions,
        y=target,
        cif_id=cif_id,
        atom_num=atom_num,
        max_nbrs=max_nbrs,
        )
            
    @staticmethod
    def collate_fn(data_list):
        """
        Collate function for PyTorch Geometric DataLoader.

        Batches a list of `torch_geometric.data.Data` objects into a single `Batch`.

        Args:
            data_list (List[Data]): List of PyG Data objects (e.g., one per crystal)

        Returns:
            Batch: A PyG Batch object containing all the fields (x, edge_index, y, etc.)
        """
        return Batch.from_data_list(data_list)
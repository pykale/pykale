import os
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

from kale.evaluate.metrics import GaussianDistance
from kale.prepdata.materials_features import AtomCustomJSONInitializer


class CIFData(Dataset):
    """
    Dataset for loading CIF files and extracting features for crystal structures.

    Args:
        mpids_bg (pd.DataFrame): DataFrame containing material IDs and bandgap values.
        cif_folder (str): Path to the folder containing CIF files.
        init_file (str): Path to the JSON file for atom feature initialization.
        max_nbrs (int): Maximum number of neighbors to consider for each atom.
        radius (float): Radius for neighbor search.
        randomize (bool): Whether to randomize the dataset order.
        dmin (float, optional): Minimum distance for Gaussian distance calculation. Default is 0.
        step (float, optional): Step size for Gaussian distance calculation. Default is 0.2.
    """

    def __init__(self, mpids_bg, cif_folder, init_file, max_nbrs, radius, randomize, dmin=0, step=0.2):
        self.max_num_nbr, self.radius = max_nbrs, radius

        if randomize:
            self.mpids_bg_dataset = (
                mpids_bg.sample(frac=1).reset_index(drop=True).values
            )  # Shuffling data and converting df to array
        else:
            self.mpids_bg_dataset = mpids_bg.reset_index(drop=True).values

        atom_init_file = init_file
        self.cif_folder = cif_folder
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    @staticmethod
    def collate_fn(data_list):
        batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
        batch_positions, batch_atom_num = [], []
        crystal_atom_idx, batch_target = [], []
        batch_cif_ids = []
        batch_atom_indices = []
        base_idx = 0

        for i, data_item in enumerate(data_list):
            n_i = data_item.atom_fea.shape[0]  # number of atoms in this crystal

            # append features
            batch_atom_fea.append(data_item.atom_fea)
            batch_nbr_fea.append(data_item.nbr_fea)
            batch_nbr_fea_idx.append(data_item.nbr_fea_idx + base_idx)
            batch_positions.append(data_item.positions)
            batch_atom_num.append(data_item.atom_num)

            # index mappings
            crystal_atom_idx.append(torch.arange(n_i, device=data_item.atom_fea.device) + base_idx)
            batch_target.append(data_item.target)
            batch_cif_ids.append(data_item.cif_id)
            batch_atom_indices.append(torch.full((n_i,), i, dtype=torch.long, device=data_item.atom_fea.device))

            base_idx += n_i

        return SimpleNamespace(
            atom_fea=torch.cat(batch_atom_fea),
            nbr_fea=torch.cat(batch_nbr_fea),
            nbr_fea_idx=torch.cat(batch_nbr_fea_idx),
            positions=torch.cat(batch_positions),
            atom_num=torch.cat(batch_atom_num),
            crystal_atom_idx=crystal_atom_idx,
            target=torch.stack(batch_target),
            cif_ids=batch_cif_ids,
            batch_idx=torch.cat(batch_atom_indices),
            batch_size=len(batch_cif_ids),
        )

    def __len__(self):
        return len(self.mpids_bg_dataset)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target_np = self.mpids_bg_dataset[idx]
        crystal = Structure.from_file(os.path.join(self.cif_folder, cif_id + ".cif"))

        Z = np.asarray(crystal.atomic_numbers, dtype=np.int64)
        atom_fea = torch.tensor(np.vstack([self.ari.get_atom_fea(z) for z in Z]), dtype=torch.float32)

        target = torch.Tensor([float(target_np)])
        positions = torch.Tensor(crystal.cart_coords)
        atom_num = torch.Tensor(crystal.atomic_numbers).long()

        all_nbrs = crystal.get_all_neighbors(self.radius)  # include_index is depreciated. Index is always included now.
        all_nbrs = [
            sorted(nbrs, key=lambda x: x.nn_distance) for nbrs in all_nbrs
        ]  # Sorts nbrs based on the value of key as applied to each element of the list.
        nbr_fea_idx, nbr_fea_np = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id)
                )
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_np.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.0] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x.index, nbr[: self.max_num_nbr])))
                nbr_fea_np.append(list(map(lambda x: x.nn_distance, nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea_np = np.array(nbr_fea_idx), np.array(nbr_fea_np)
        nbr_fea_gp = self.gdf.expand(nbr_fea_np)

        nbr_fea = torch.Tensor(nbr_fea_gp)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return CIFDataItem(atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, target, cif_id)


class CIFDataItem:
    def __init__(self, atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, target, cif_id):
        self.atom_fea = atom_fea
        self.nbr_fea = nbr_fea
        self.nbr_fea_idx = nbr_fea_idx
        self.positions = positions
        self.atom_num = atom_num
        self.target = target
        self.cif_id = cif_id

    def to_dict(self):
        return {
            "atom_fea": self.atom_fea,
            "nbr_fea": self.nbr_fea,
            "nbr_fea_idx": self.nbr_fea_idx,
            "positions": self.positions,
            "atom_num": self.atom_num,
            "target": self.target,
            "cif_id": self.cif_id,
        }

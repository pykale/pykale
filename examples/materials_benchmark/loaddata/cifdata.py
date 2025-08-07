# import json
# import os
# import warnings

# import torch
# import numpy as np
# from pymatgen.core import Structure
# from torch.utils.data import Dataset
# from torch_geometric.data import Data, Batch



# class GaussianDistance(object):
#     """
#     Expands the distance by Gaussian basis.

#     Unit: angstrom
#     """
#     def __init__(self, dmin, dmax, step, var=None):
#         """
#         Parameters
#         ----------

#         dmin: float
#           Minimum interatomic distance
#         dmax: float
#           Maximum interatomic distance
#         step: float
#           Step size for the Gaussian filter
#         """
#         assert dmin < dmax
#         assert dmax - dmin > step
#         self.filter = np.arange(dmin, dmax+step, step)
#         if var is None:
#             var = step
#         self.var = var

#     def expand(self, distances):
#         """
#         Apply Gaussian disntance filter to a numpy distance array

#         Parameters
#         ----------

#         distance: np.array shape n-d array
#           A distance matrix of any shape

#         Returns
#         -------
#         expanded_distance: shape (n+1)-d array
#           Expanded distance matrix with the last dimension of length
#           len(self.filter)
#         """
#         return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
#                       self.var**2)


# class AtomInitializer(object):
#     """
#     Base class for intializing the vector representation for atoms.

#     !!! Use one AtomInitializer per dataset !!!
#     """
#     def __init__(self, atom_types):
#         self.atom_types = set(atom_types)
#         self._embedding = {}

#     def get_atom_fea(self, atom_type):
#         assert atom_type in self.atom_types
#         return self._embedding[atom_type]

#     def load_state_dict(self, state_dict):
#         self._embedding = state_dict
#         self.atom_types = set(self._embedding.keys())
#         self._decodedict = {idx: atom_type for atom_type, idx in
#                             self._embedding.items()}

#     def state_dict(self):
#         return self._embedding

#     def decode(self, idx):
#         if not hasattr(self, '_decodedict'):
#             self._decodedict = {idx: atom_type for atom_type, idx in
#                                 self._embedding.items()}
#         return self._decodedict[idx]


# class AtomCustomJSONInitializer(AtomInitializer):
#     """
#     Initialize atom feature vectors using a JSON file, which is a python
#     dictionary mapping from element number to a list representing the
#     feature vector of the element.

#     Parameters
#     ----------

#     elem_embedding_file: str
#         The path to the .json file
#     """
#     def __init__(self, elem_embedding_file):
#         with open(elem_embedding_file) as f:
#             elem_embedding = json.load(f)
#         elem_embedding = {int(key): value for key, value
#                           in elem_embedding.items()}
#         atom_types = set(elem_embedding.keys())
#         super(AtomCustomJSONInitializer, self).__init__(atom_types)
#         for key, value in elem_embedding.items():
#             self._embedding[key] = np.array(value, dtype=float)





# class CIFDataItem:
#     """
#     A data container representing a single crystal graph sample.

#     This class wraps all the necessary information extracted from a CIF file
#     for downstream use in crystal graph neural networks.

#     Attributes:
#         atom_fea (Tensor): Tensor of shape (N_atoms, F) containing atom-level features.
#         nbr_fea (Tensor): Tensor of shape (N_atoms, max_nbrs, F') containing neighbor distances/features.
#         nbr_fea_idx (LongTensor): Tensor of shape (N_atoms, max_nbrs) indicating neighbor indices.
#         positions (Tensor): Tensor of shape (N_atoms, 3) for atomic Cartesian coordinates.
#         atom_num (LongTensor): Tensor of shape (N_atoms,) with atomic numbers.
#         target (Tensor): Tensor of shape (1,) representing the target property (e.g., band gap).
#         cif_id (str): Unique ID (e.g., MPID) for the CIF file.

#     Note:
#         This object is returned by `CIFData.__getitem__()` and consumed by `collate_fn`
#         for batched model input.
#     """
#     def __init__(self, atom_fea, nbr_fea, nbr_fea_idx, positions, atom_num, target, cif_id):
#         self.atom_fea = atom_fea
#         self.nbr_fea = nbr_fea
#         self.nbr_fea_idx = nbr_fea_idx
#         self.positions = positions
#         self.atom_num = atom_num
#         self.target = target
#         self.cif_id = cif_id

#     def to_dict(self):
#         return {
#             "atom_fea": self.atom_fea,
#             "nbr_fea": self.nbr_fea,
#             "nbr_fea_idx": self.nbr_fea_idx,
#             "positions": self.positions,
#             "atom_num": self.atom_num,
#             "target": self.target,
#             "cif_id": self.cif_id
#         }
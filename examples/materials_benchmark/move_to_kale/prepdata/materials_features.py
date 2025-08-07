import numpy as np
import json

def extract_features(dataset):

    """
    Extracts fixed-size features from PyG-style crystal graph data for use in non-GNN models.

    For each sample, this function:
        - collects atom features (x), neighbor features (edge_attr), and neighbor indices (edge_index)
        - reconstructs atom–neighbor interaction matrices
        - aggregates over neighbors and atoms to form a graph-level vector

    Args:
        dataset (Iterable[torch_geometric.data.Data]): list or dataset of PyG Data objects

    Returns:
        features (np.ndarray): shape (n_samples, n_features)
        targets (np.ndarray): shape (n_samples,)
    """

    features, targets = [], []
    for data in dataset:
        atom_fea = data.atom_fea.cpu().numpy()  # (N_atoms, atom_fea_len)
        nbr_fea = data.nbr_fea.cpu().numpy()  # (N_atoms, N_neighbors, nbr_fea_len)
        nbr_fea_idx = data.nbr_fea_idx.cpu().numpy()  # (N_atoms, N_neighbors)
        # Get neighbor atom features
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_fea[nbr_fea_idx]  # (N_atoms, N_neighbors, atom_fea_len)
        # Expand atom_fea to match dimensions
        atom_fea = np.expand_dims(atom_fea, axis=1)  # (N_atoms, 1, feature_dim)
        atom_fea = np.tile(atom_fea, (1, M, 1))  # (N_atoms, N_neighbors, feature_dim)

        # Concatenate atom and neighbor features
        total_fea = np.concatenate([atom_fea, atom_nbr_fea, nbr_fea], axis=2)  # (N_atoms, N_neighbors, input_dim)

        # Aggregate over neighbors
        total_fea = np.mean(total_fea, axis=1)  # (N_atoms, input_dim)
        # total_fea = np.concatenate([atom_fea, total_fea], axis=-1)  # (N_atoms, input_dim + atom_fea_len)
        total_fea = np.mean(total_fea, axis=0)



        # Store features and target (bandgap value)
        features.append(total_fea)
        targets.append(data.target.cpu().numpy())  # Bandgap target

    features = np.vstack(features)  # (Total samples, Feature size)
    targets = np.concatenate(targets)  # (Total samples,)
    return features, targets


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

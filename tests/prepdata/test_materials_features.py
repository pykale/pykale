import json
import types

import numpy as np
import pytest
import torch

from kale.prepdata.materials_features import AtomCustomJSONInitializer, AtomInitializer, extract_features


def test_extract_features():
    # atom_fea: (N, A)
    # nbr_fea: (N, M, B)
    # nbr_fea_idx: (N, M)
    # target: (1,)
    # make a sample
    atom_fea1 = torch.full((2, 3), float(1), dtype=torch.float32)  # 2 atoms, 3-dim features
    # neighbor features (constant)
    nbr_fea1 = torch.full((2, 2, 2), float(2), dtype=torch.float32)  # 2 atoms, 2 neighbors, 2-dim features
    # simple valid neighbor indices: [[0,1],[1,0]]
    idx1 = np.vstack([np.arange(2), np.arange(2)[::-1]]).T  # shape (2,2) for n_atoms==2
    nbr_fea_idx1 = torch.tensor(idx1, dtype=torch.long)
    # target as 1D tensor of length 1
    target1 = torch.tensor([float(0.5)], dtype=torch.float32)  # target as 1D tensor of length 1

    sample1 = types.SimpleNamespace(
        atom_fea=atom_fea1,
        nbr_fea=nbr_fea1,
        nbr_fea_idx=nbr_fea_idx1,
        target=target1,
    )

    atom_fea2 = torch.full((3, 3), float(3), dtype=torch.float32)  # 3 atom, 3-dim features
    nbr_fea2 = torch.full((3, 2, 2), float(2), dtype=torch.float32)  # 3 atoms, 2 neighbors, 2-dim features
    # simple valid neighbor indices: [[0,1],[1,2],[2,0]]
    idx2 = np.vstack([np.arange(3), (np.arange(3) + 1) % 3]).T  # shape (3,2) for n_atoms==3
    nbr_fea_idx2 = torch.tensor(idx2, dtype=torch.long)
    target2 = torch.tensor([-1.0], dtype=torch.float32)  # target as 1D tensor of length 1
    sample2 = types.SimpleNamespace(
        atom_fea=atom_fea2,
        nbr_fea=nbr_fea2,
        nbr_fea_idx=nbr_fea_idx2,
        target=target2,
    )

    dataset = [sample1, sample2]
    x, y = extract_features(dataset)

    # ---- shape checks ----
    # Feature length = 2*atom_dim + edge_dim = 2*3 + 2 = 8
    assert x.shape == (2, 8)
    assert y.shape == (2,)

    # ---- value checks (because we used constants, the averages are trivial) ----
    # Each feature vector should be: [atom]*A + [atom_of_neighbor]*A + [edge]*B
    # With constants, atom == neighbor_atom == atom_val, so:
    # expected = [atom_val]*A + [atom_val]*A + [edge_val]*B
    expected_s1 = np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.float32)
    expected_s2 = np.array([3, 3, 3, 3, 3, 3, 2, 2], dtype=np.float32)

    np.testing.assert_allclose(x[0], expected_s1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[1], expected_s2, rtol=0, atol=1e-6)

    # targets preserved in order
    np.testing.assert_allclose(y, np.array([0.5, -1.0], dtype=np.float32), rtol=0, atol=1e-6)


def test_atom_initializer():
    # Base initializer expects an atom->index mapping (integers), not vectors.
    ai = AtomInitializer(atom_types={1, 6})
    state = {1: 42, 6: 99}
    ai.load_state_dict(state)

    # get_atom_fea returns the mapped value
    assert ai.get_atom_fea(1) == 42
    assert ai.get_atom_fea(6) == 99

    # state roundtrip
    assert ai.state_dict() == state

    # decode should invert the mapping
    assert ai.decode(42) == 1
    assert ai.decode(99) == 6

    # unknown atom type should assert
    with pytest.raises(AssertionError):
        ai.get_atom_fea(8)


def test_atom_custom_json_initializer(tmp_path):
    # Create a tiny JSON with string keys (as typical JSON) -> class should convert to int keys
    payload = {"1": [0.1, 0.2], "6": [0.3, 0.4]}
    json_path = tmp_path / "atom_init.json"
    with open(json_path, "w") as f:
        json.dump(payload, f)

    ajson = AtomCustomJSONInitializer(str(json_path))

    # Keys converted to ints
    assert ajson.atom_types == {1, 6}

    # get_atom_fea returns numpy arrays of dtype float
    fea_1 = ajson.get_atom_fea(1)
    fea_6 = ajson.get_atom_fea(6)
    assert isinstance(fea_1, np.ndarray) and fea_1.dtype == float
    assert isinstance(fea_6, np.ndarray) and fea_6.dtype == float

    np.testing.assert_allclose(fea_1, np.array([0.1, 0.2], dtype=float))
    np.testing.assert_allclose(fea_6, np.array([0.3, 0.4], dtype=float))

    # state_dict returns the mapping of int->np.array
    state = ajson.state_dict()
    assert set(state.keys()) == {1, 6}
    np.testing.assert_allclose(state[1], [0.1, 0.2])
    np.testing.assert_allclose(state[6], [0.3, 0.4])

    # unknown atom type should assert
    with pytest.raises(AssertionError):
        ajson.get_atom_fea(8)

# conftest or test file
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import DataLoader

from kale.loaddata.materials_datasets import CIFData, CrystalDataset


@pytest.fixture(scope="module")
def dummy_data():
    temp_dir = tempfile.TemporaryDirectory()
    paths = create_dummy_data(temp_dir.name)
    yield paths
    temp_dir.cleanup()


def create_dummy_data(data_dir):
    """
    Create a tiny crystal dataset:
      - data_dir/cif_file/mp-1.cif (Si), mp-2.cif (NaCl)
      - data_dir/train.json         -> {"mp-1": 1.23, "mp-2": 2.34}
      - data_dir/atom_init.json     -> minimal per-element feature vectors

    Returns a dict of useful paths.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    cif_dir = root / "cif_file"
    cif_dir.mkdir(parents=True, exist_ok=True)

    si_lat = Lattice.cubic(5.43)
    si = Structure(si_lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    (cif_dir / "mp-1.cif").write_text(si.to(fmt="cif"))

    nacl_lat = Lattice.cubic(5.64)
    nacl = Structure(nacl_lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    (cif_dir / "mp-2.cif").write_text(nacl.to(fmt="cif"))

    # Labels JSON (e.g., band gap)
    labels_path = root / "train.json"
    labels = {"mp-1": 0.61, "mp-2": 5}
    labels_path.write_text(json.dumps(labels))

    # atom_init.json
    # keys are atomic numbers as strings
    atom_init_path = root / "atom_init.json"
    atom_init = {
        "11": [1, 0, 0],  # Na
        "14": [0, 1, 0],  # Si
        "17": [0, 0, 1],  # Cl
    }
    atom_init_path.write_text(json.dumps(atom_init))

    return {
        "data_dir": str(root),
        "cif_folder": str(cif_dir),
        "target_path": str(labels_path),
        "init_file": str(atom_init_path),
    }


def test_cifdata(tmp_path):
    # ---------- 1) atom_init.json (Na=11, Cl=17 -> 2-dim features) ----------
    init_file = tmp_path / "atom_init.json"
    with open(init_file, "w") as f:
        json.dump({"11": [0.0, 1.0], "17": [1.0, 0.5]}, f)

    # ---------- 2) two tiny CIFs (rock-salt NaCl) ----------
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    lat = Lattice.cubic(4.0)
    struct = Structure(
        lattice=lat,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        coords_are_cartesian=False,
    )
    for name in ["mat1", "mat2"]:
        struct.to(fmt="cif", filename=str(cif_dir / f"{name}.cif"))

    # ---------- 3) DataFrame: mpids + targets ----------
    df = pd.DataFrame({"mpids": ["mat1", "mat2"], "bg": [1.23, 2.34]})

    # common params
    max_nbrs = 4
    radius = 6.0  # Å
    expected_rbf_len = len(np.arange(0.0, radius + 0.2, 0.2))  # matches GaussianDistance(dmin=0, step=0.2)

    # ---------- 4) CIFData (deterministic order) ----------
    ds = CIFData(
        mpids_bg=df[["mpids", "bg"]],
        cif_folder=str(cif_dir),
        init_file=str(init_file),
        max_nbrs=max_nbrs,
        radius=radius,
        randomize=False,
    )

    # length
    assert len(ds) == 2

    # first sample
    s0 = ds[0]
    assert isinstance(s0.atom_fea, torch.Tensor) and s0.atom_fea.shape == (2, 2)  # 2 atoms, 2-dim features
    assert isinstance(s0.positions, torch.Tensor) and s0.positions.shape == (2, 3)
    assert isinstance(s0.atom_num, torch.Tensor) and s0.atom_num.shape == (2,)
    assert isinstance(s0.nbr_fea, torch.Tensor) and s0.nbr_fea.shape == (2, max_nbrs, expected_rbf_len)
    assert isinstance(s0.nbr_fea_idx, torch.Tensor) and s0.nbr_fea_idx.shape == (2, max_nbrs)
    assert isinstance(s0.target, torch.Tensor) and s0.target.shape == (1,)
    assert isinstance(s0.cif_id, str) and s0.cif_id == "mat1"

    # collate two samples
    batch = CIFData.collate_fn([ds[0], ds[1]])
    assert batch.atom_fea.shape == (4, 2)  # 2 crystals * 2 atoms
    assert batch.nbr_fea.shape[0] == 4
    assert batch.nbr_fea.shape[1] == max_nbrs
    assert batch.nbr_fea.shape[2] == expected_rbf_len
    assert batch.positions.shape == (4, 3)
    assert isinstance(batch.crystal_atom_idx, list) and len(batch.crystal_atom_idx) == 2
    assert batch.target.shape == (2, 1)
    assert batch.batch_size == 2
    assert len(batch.cif_ids) == 2 and batch.cif_ids[0] == "mat1" and batch.cif_ids[1] == "mat2"

    # DataLoader one batch
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=CIFData.collate_fn)
    b = next(iter(loader))
    assert b.atom_fea.shape[0] == 4 and b.batch_size == 2

    # ---------- 5) CIFData (randomize=True) preserves set of ids ----------
    ds_shuf = CIFData(
        mpids_bg=df[["mpids", "bg"]],
        cif_folder=str(cif_dir),
        init_file=str(init_file),
        max_nbrs=max_nbrs,
        radius=radius,
        randomize=True,
    )
    # order may or may not change, but set of IDs must match
    ids_orig = list(df["mpids"])
    ids_shuf = [row[0] for row in ds_shuf.mpids_bg_dataset]
    assert set(ids_orig) == set(ids_shuf)


def test_crystal_dataset(tmp_path):
    # --- 1) Write minimal atom_init.json (Na=11, Cl=17 -> 2-dim embedding) ---
    init_file = tmp_path / "atom_init.json"
    with open(init_file, "w") as f:
        json.dump({"11": [0.0, 1.0], "17": [1.0, 0.5]}, f)

    # --- 2) Write three tiny CIFs (NaCl rock-salt) ---
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    lat = Lattice.cubic(4.0)
    s = Structure(lattice=lat, species=["Na", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]], coords_are_cartesian=False)
    for name in ["mat1", "mat2", "mat3"]:
        s.to(fmt="cif", filename=str(cif_dir / f"{name}.cif"))

    # --- 3) Build DFs (train/val/test) ---
    train_df = pd.DataFrame({"mpids": ["mat1", "mat2"], "bg": [1.1, 2.2]})
    val_df = pd.DataFrame({"mpids": ["mat3"], "bg": [3.3]})
    test_df = pd.DataFrame({"mpids": ["mat2"], "bg": [2.0]})

    # common params
    batch_size = 2
    radius = 5.0
    max_nbrs = 4

    # --- 4) Case A: without test_df ---
    ds_no_test = CrystalDataset(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        cif_folder=str(cif_dir),
        init_file=str(init_file),
        max_nbrs=max_nbrs,
        radius=radius,
        target_key="bg",
        batch_size=batch_size,
        num_workers=0,
        randomize_train=False,
    )
    train_loader = ds_no_test.get_train_loader(shuffle=False)
    val_loader = ds_no_test.get_valid_loader()
    test_loader = ds_no_test.get_test_loader()
    assert test_loader is None

    # grab one batch from train
    batch = next(iter(train_loader))
    # basic fields exist
    for attr in [
        "atom_fea",
        "nbr_fea",
        "nbr_fea_idx",
        "positions",
        "atom_num",
        "target",
        "crystal_atom_idx",
        "cif_ids",
        "batch_idx",
        "batch_size",
    ]:
        assert hasattr(batch, attr)
    # dims: atom features length == 2 (from atom_init.json)
    assert batch.atom_fea.shape[1] == 2
    # neighbors second dim equals max_nbrs
    assert batch.nbr_fea.shape[1] == max_nbrs
    assert batch.nbr_fea_idx.shape[1] == max_nbrs
    # positions are 3D
    assert batch.positions.shape[1] == 3
    # targets per-graph equals batch_size
    assert batch.target.shape[0] == batch.batch_size
    # crystal_atom_idx list length == batch_size
    assert isinstance(batch.crystal_atom_idx, list) and len(batch.crystal_atom_idx) == batch.batch_size
    # batch_idx covers all atoms
    assert batch.batch_idx.numel() == batch.atom_fea.shape[0]

    # feature_dims() check
    a_len, e_len, p_len = ds_no_test.feature_dims()
    assert a_len == 2
    expected_rbf = len(np.arange(0.0, radius + 0.2, 0.2))  # matches GaussianDistance(dmin=0, step=0.2)
    assert e_len == expected_rbf
    assert p_len == 3

    # val loader yields one batch
    val_batch = next(iter(val_loader))
    assert val_batch.target.shape[0] == val_batch.batch_size

    # --- 5) Case B: with test_df ---
    ds_with_test = CrystalDataset(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cif_folder=str(cif_dir),
        init_file=str(init_file),
        max_nbrs=max_nbrs,
        radius=radius,
        target_key="bg",
        batch_size=batch_size,
        num_workers=0,
        randomize_train=False,
    )
    assert ds_with_test.get_test_loader() is not None

    # --- 6) Column validation error ---
    bad_df = pd.DataFrame({"mpids": ["mat1", "mat2"]})  # missing 'bg'
    with pytest.raises(ValueError):
        _ = CrystalDataset(
            train_df=bad_df,
            val_df=val_df,
            test_df=test_df,
            cif_folder=str(cif_dir),
            init_file=str(init_file),
            max_nbrs=max_nbrs,
            radius=radius,
            target_key="bg",
        )

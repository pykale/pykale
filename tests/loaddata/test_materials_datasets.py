# conftest or test file
import json
import tempfile
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from torch.utils.data import DataLoader

from kale.loaddata.materials_datasets import CIFData


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


def test_cifdata(dummy_data):
    dataset = CIFData(
        target_path=dummy_data["target_path"],
        cif_folder=dummy_data["cif_folder"],
        init_file=dummy_data["init_file"],
        max_nbrs=12,
        radius=8.0,
        randomize=False,
        target_key="bg",
    )

    assert len(dataset) == 2

    assert dataset.load_data(dummy_data["target_path"], "bg").shape == (2, 2)

    item = dataset[0]
    assert item.atom_fea.shape[1] == 3
    assert item.nbr_fea.shape[1] == 12
    assert item.nbr_fea_idx.shape[1] == 12
    assert item.positions.shape[1] == 3
    assert item.atom_num.shape[0] == item.atom_fea.shape[0]
    assert isinstance(item.cif_id, str)
    assert item.target.shape == (1,)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=CIFData.collate_fn, num_workers=0)
    batch = next(iter(loader))
    assert batch.batch_size == 2
    assert hasattr(batch, "atom_fea") and hasattr(batch, "nbr_fea_idx") and hasattr(batch, "positions")
    assert (
        batch.atom_fea.shape[0]
        == batch.nbr_fea.shape[0]
        == batch.positions.shape[0]
        == batch.atom_num.shape[0]
        == batch.batch_idx.shape[0]
    )
    assert batch.target.shape == (2, 1)

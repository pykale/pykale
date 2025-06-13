import numpy as np
import pandas as pd
import pytest
import torch

from kale.loaddata.signal_access import load_ecg_from_dir


@pytest.fixture
def make_csv(tmp_path):
    def _make(file_names):
        csv_path = tmp_path / "signals.csv"
        df = pd.DataFrame({"path": file_names})
        df.to_csv(csv_path, index=False)
        return csv_path

    return _make


@pytest.mark.parametrize("bad_shape", [False, True])
def test_load_ecg_from_dir(tmp_path, monkeypatch, make_csv, bad_shape):
    # -- Setup
    file_names = ["a1.dat", "a2.dat"]
    csv_path = make_csv(file_names)
    root = str(tmp_path)

    # Dummy signal: 10 samples x 2 channels
    dummy_wave = torch.arange(20).reshape(10, 2).numpy()
    dummy_meta = {"n_sig": 2}

    # If bad_shape, fudge the array to the wrong shape to force the skip code path
    if bad_shape:
        dummy_wave = np.arange(22)
        dummy_meta = {"n_sig": 3}

    # Patch dependencies
    monkeypatch.setattr("kale.loaddata.signal_access.wfdb.rdsamp", lambda f: (dummy_wave, dummy_meta))
    monkeypatch.setattr("kale.loaddata.signal_access.interpolate_signal", lambda s: s)
    monkeypatch.setattr("kale.loaddata.signal_access.normalize_signal", lambda s: s)

    # Prepare ecg tensor always returns shape (1, 1, total_samples)
    def dummy_prepare_ecg_tensor(signal):
        n = signal.shape[0] * signal.shape[1]
        return torch.ones(1, 1, n)

    monkeypatch.setattr("kale.loaddata.signal_access.prepare_ecg_tensor", dummy_prepare_ecg_tensor)

    # -- Test
    batch = load_ecg_from_dir(root, csv_path.name)
    if not bad_shape:
        # Should stack both
        assert batch.shape == (2, 1, 20)
        assert torch.all(batch == 1)
    else:
        # Both fail and should get empty tensor
        assert batch.shape == (0,)

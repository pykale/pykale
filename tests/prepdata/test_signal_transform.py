import numpy as np
import pytest
import torch

from kale.prepdata.signal_transform import interpolate_signal, normalize_signal, prepare_ecg_tensor


def test_normalize_signal_regular():
    # Shape: (samples, channels)
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    normed = normalize_signal(arr)
    assert normed.shape == arr.shape
    # Mean should be zero after normalization (numerical tolerance)
    assert np.allclose(np.mean(normed, axis=0), 0, atol=1e-6)
    # Std should be one after normalization
    assert np.allclose(np.std(normed, axis=0), 1, atol=1e-6)


def test_normalize_signal_zero_std():
    # All same values in one channel
    arr = np.array([[1.0, 2.0], [1.0, 2.0]])
    normed = normalize_signal(arr)
    # No division by zero, still finite
    assert np.isfinite(normed).all()
    # The column with constant value should be all zeros
    assert np.all(normed[:, 0] == 0)
    assert np.all(normed[:, 1] == 0)


def test_interpolate_signal_nan_middle():
    arr = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 6.0]])
    interpolated = interpolate_signal(arr)
    assert interpolated.shape == arr.shape
    # Should fill the nan as average of neighbors
    assert np.allclose(interpolated[1], [2.0, 4.0], atol=1e-6)


def test_interpolate_signal_no_nan():
    arr = np.arange(6).reshape(3, 2)
    interpolated = interpolate_signal(arr)
    assert np.allclose(arr, interpolated)


def test_prepare_ecg_tensor_numpy():
    arr = np.arange(6).reshape(3, 2)
    tensor = prepare_ecg_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 6)
    assert tensor.dtype == torch.float32
    # Values preserved and order correct
    assert torch.allclose(tensor, torch.tensor(arr.reshape(1, -1), dtype=torch.float32))


def test_prepare_ecg_tensor_tensor():
    arr = torch.arange(6).reshape(3, 2)
    tensor = prepare_ecg_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 6)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, arr.reshape(1, -1).to(torch.float32))


def test_prepare_ecg_tensor_empty_numpy():
    arr = np.array([]).reshape(0, 2)
    tensor = prepare_ecg_tensor(arr)
    assert tensor.shape == (1, 0)
    assert tensor.dtype == torch.float32


def test_prepare_ecg_tensor_empty_tensor():
    arr = torch.empty(0, 2)
    tensor = prepare_ecg_tensor(arr)
    assert tensor.shape == (1, 0)
    assert tensor.dtype == torch.float32


def test_prepare_ecg_tensor_wrong_type():
    with pytest.raises(TypeError):
        prepare_ecg_tensor([1, 2, 3])  # List, not supported

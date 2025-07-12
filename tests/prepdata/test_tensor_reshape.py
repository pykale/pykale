import torch

from kale.prepdata.tensor_reshape import normalize_tensor


def test_normalize_basic():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    norm = normalize_tensor(tensor)
    assert torch.allclose(norm, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


def test_normalize_negative_values():
    tensor = torch.tensor([-2.0, 0.0, 2.0])
    norm = normalize_tensor(tensor)
    assert torch.allclose(norm, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


def test_normalize_all_same_values():
    tensor = torch.tensor([5.0, 5.0, 5.0])
    norm = normalize_tensor(tensor)
    assert torch.allclose(norm, torch.tensor([0.0, 0.0, 0.0]), atol=1e-5)


def test_normalize_2d_tensor():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm = normalize_tensor(tensor)
    expected = (tensor - 1.0) / (4.0 - 1.0 + 1e-8)
    assert torch.allclose(norm, expected, atol=1e-6)

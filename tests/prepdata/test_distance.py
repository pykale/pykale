import pytest
import torch

from kale.utils.distance import calculate_distance, DistanceMetric


@pytest.fixture
def x1():
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)


@pytest.fixture
def x2():
    return torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)


def test_cosine_distance(x1, x2):
    expected_output = torch.tensor([[0.9258, 0.9258], [0.9869, 0.9869]])
    output = calculate_distance(x1, x2, metric=DistanceMetric.COSINE)
    assert torch.allclose(output, expected_output, rtol=1e-3, atol=1e-3)


def test_cosine_distance_self(x1):
    expected_output = torch.tensor([[1.0000, 0.9746], [0.9746, 1.0000]])
    output = calculate_distance(x1, metric=DistanceMetric.COSINE)
    assert torch.allclose(output, expected_output, rtol=1e-3, atol=1e-3)


def test_cosine_distance_eps(x1, x2):
    expected_output = torch.tensor([[0.9258, 0.9258], [0.9869, 0.9869]])
    output = calculate_distance(x1, x2, eps=0, metric=DistanceMetric.COSINE)
    assert torch.allclose(output, expected_output, rtol=1e-3, atol=1e-3)


def test_unsupported_metric(x1, x2):
    with pytest.raises(Exception):
        calculate_distance(x1, x2, metric="unsupported_metric")

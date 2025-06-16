import random

import pytest
import torch

from kale.evaluate.metrics import (
    binary_cross_entropy,
    calculate_distance,
    DistanceMetric,
    multitask_topk_accuracy,
    protonet_loss,
    signal_image_elbo_loss,
    topk_accuracy,
)

# Dummy data: [batch_size, num_classes]
# Dummy ground truth: batch_size
FIRST_PREDS = torch.tensor(
    (
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
)
FIRST_LABELS = torch.tensor((0, 2, 4, 5, 5))

SECOND_PREDS = torch.tensor(
    (
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
)
SECOND_LABELS = torch.tensor((0, 0, 4, 4, 5))

MULTI_PREDS = (FIRST_PREDS, SECOND_PREDS)
MULTI_LABELS = (FIRST_LABELS, SECOND_LABELS)


def test_topk_accuracy():
    # Test topk_accuracy with single-task input
    preds = FIRST_PREDS
    labels = FIRST_LABELS
    k = (1, 3, 5)

    top1, top3, top5 = topk_accuracy(preds, labels, k)
    top1_value = top1.double().mean()
    top3_value = top3.double().mean()
    top5_value = top5.double().mean()
    assert top1_value.cpu() == pytest.approx(1 / 5)
    assert top3_value.cpu() == pytest.approx(2 / 5)
    assert top5_value.cpu() == pytest.approx(3 / 5)


def test_multitask_topk_accuracy():
    # Test multitask_topk_accuracy with input for two tasks
    preds = MULTI_PREDS
    labels = MULTI_LABELS
    k = (1, 3, 5)

    top1, top3, top5 = multitask_topk_accuracy(preds, labels, k)
    top1_value = top1.double().mean()
    top3_value = top3.double().mean()
    top5_value = top5.double().mean()
    assert top1_value.cpu() == pytest.approx(1 / 5)
    assert top3_value.cpu() == pytest.approx(2 / 5)
    assert top5_value.cpu() == pytest.approx(3 / 5)


def test_protonet_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = random.randint(1, 10)
    num_support_samples = random.randint(1, 10)
    num_query_samples = random.randint(1, 10)
    n_dim = random.randint(1, 512)
    feature_support = torch.rand(num_classes, num_support_samples, n_dim)
    feature_query = torch.rand(num_classes * num_query_samples, n_dim)
    loss_fn = protonet_loss(num_classes=num_classes, num_query_samples=num_query_samples, device=device)
    loss, acc = loss_fn(feature_support, feature_query)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(acc, torch.Tensor)
    prototypes = feature_support.mean(dim=1)
    dists = loss_fn.euclidean_dist_for_tensor_group(prototypes, feature_query)
    assert dists.shape == (num_classes, num_classes * num_query_samples)


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


@pytest.mark.parametrize(
    "use_image, use_signal",
    [
        (True, True),  # Both modalities present
        (True, False),  # Only image present
        (False, True),  # Only signal present
        (False, False),  # Neither present (degenerate)
    ],
)
def test_signal_image_elbo_loss_branches(use_image, use_signal):
    batch_size = 4
    latent_dim = 8
    image_shape = (batch_size, 3, 2, 2)
    signal_shape = (batch_size, 1, 2, 2)

    recon_image = torch.randn(image_shape) if use_image else None
    target_image = torch.randn(image_shape) if use_image else None
    recon_signal = torch.randn(signal_shape) if use_signal else None
    target_signal = torch.randn(signal_shape) if use_signal else None

    mean = torch.zeros(batch_size, latent_dim)
    log_var = torch.zeros(batch_size, latent_dim)

    loss = signal_image_elbo_loss(
        recon_image,
        target_image,
        recon_signal,
        target_signal,
        mean,
        log_var,
        lambda_image=2.0,
        lambda_signal=3.0,
        annealing_factor=0.7,
        scale_factor=1e-3,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 or loss.numel() == 1  # Scalar loss
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"


def test_signal_image_elbo_loss_values():
    """Test signal_image_elbo_loss with known values for a small example."""
    batch_size = 2
    latent_dim = 2
    recon_image = torch.zeros(batch_size, 2)
    target_image = torch.ones(batch_size, 2)
    recon_signal = torch.zeros(batch_size, 2)
    target_signal = torch.ones(batch_size, 2)
    mean = torch.zeros(batch_size, latent_dim)
    log_var = torch.zeros(batch_size, latent_dim)

    # MSE = mean((0-1)^2) = 1 per element, 4 elements per modality
    # Reduction='sum' = 4.0
    expected_recon = (1.0 * 4 + 1.0 * 4) * 1e-4  # scale_factor

    loss = signal_image_elbo_loss(
        recon_image,
        target_image,
        recon_signal,
        target_signal,
        mean,
        log_var,
        lambda_image=1.0,
        lambda_signal=1.0,
        annealing_factor=1.0,
        scale_factor=1e-4,
    )
    assert torch.isclose(loss, torch.tensor(expected_recon), atol=1e-6)


@pytest.mark.parametrize(
    "output,target",
    [
        # Standard case: binary classification with batch size 2
        (torch.tensor([[0.5], [1.0]], dtype=torch.float32), torch.tensor([0.0, 1.0], dtype=torch.float32)),
        # Edge case: outputs very close to 0 and 1
        (torch.tensor([[10.0], [-10.0]], dtype=torch.float32), torch.tensor([1.0, 0.0], dtype=torch.float32)),
        # Batch size 1
        (torch.tensor([[0.7]], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)),
        # Batch size >1 with multi-dimensional output
        (torch.tensor([[0.2], [0.8], [0.4]], dtype=torch.float32), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)),
    ],
)
def test_binary_cross_entropy(output, target):
    n, loss = binary_cross_entropy(output, target)

    # Assert output shape and value constraints
    assert isinstance(n, torch.Tensor)
    assert n.shape == target.shape
    assert torch.all((0 <= n) & (n <= 1))  # Since sigmoid

    # Assert loss is scalar and non-negative
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_binary_cross_entropy_zero_output():
    # Case where output tensor is all zeros
    output = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    target = torch.tensor([0.0, 1.0], dtype=torch.float32)
    n, loss = binary_cross_entropy(output, target)
    assert isinstance(loss, torch.Tensor)


def test_binary_cross_entropy_target_mismatch_shape():
    # Expect an error if target shape doesn't match input shape
    output = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.0]])  # Incorrect shape
    with pytest.raises(ValueError):  # Correct the expected exception type
        binary_cross_entropy(output, target)

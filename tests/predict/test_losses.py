import pytest
import torch

from kale.predict.losses import _moment_k, euclidean, multitask_topk_accuracy, topk_accuracy

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


# def test_hisc():
#     kx = torch.randn(5, 5)
#     ky = torch.randn(5, 5)
#     device = torch.device("cpu")
#     assert hsic(kx, ky, device) >= 0


def test_euclidean():
    x = torch.randn(5, 5)
    y = torch.randn(5, 5)
    assert isinstance(euclidean(x, y), torch.Tensor)
    assert euclidean(x, y) >= 0
    assert euclidean(x, x) == 0
    assert euclidean(y, y) == 0
    assert euclidean(x, y) == euclidean(y, x)


def test_moment_k():
    x = torch.randn(5, 128)
    domain_labels = torch.tensor([0, 1, 2, 3, 4])
    assert _moment_k(x, domain_labels, 2) >= 0
    assert isinstance(_moment_k(x, domain_labels, 2), torch.Tensor)

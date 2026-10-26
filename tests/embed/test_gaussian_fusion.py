"""Contracts for Gaussian parameter averaging, barycenters, and hierarchical fusion."""

import math

import pytest
import torch

from kale.embed.multimodal_fusion import (
    expert_groups,
    hierarchical_experts,
    mean_of_experts,
    mixture_of_experts,
    ProductOfExperts,
    wasserstein_barycenter,
)


def test_mixture_averages_parameters_with_per_sample_weights_and_masks():
    mean = torch.tensor([[[2.0], [10.0]], [[6.0], [20.0]]])
    log_var = torch.tensor([[[0.0], [2.0]], [[4.0], [6.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    weights = torch.tensor([[1.0, 1.0], [3.0, 5.0]])

    joint_mean, joint_log_var = mixture_of_experts(mean, log_var, mask, weights)

    torch.testing.assert_close(joint_mean, torch.tensor([[5.0], [10.0]]))
    torch.testing.assert_close(joint_log_var, torch.tensor([[3.0], [2.0]]))


def test_mean_of_experts_is_equal_weight_mixture():
    mean = torch.tensor([[[1.0]], [[3.0]], [[5.0]]])
    log_var = torch.tensor([[[-1.0]], [[0.0]], [[2.0]]])
    mask = torch.tensor([[1], [0], [1]])

    for got, expected in zip(mean_of_experts(mean, log_var, mask), mixture_of_experts(mean, log_var, mask)):
        torch.testing.assert_close(got, expected)
    torch.testing.assert_close(mean_of_experts(mean, log_var, mask)[0], torch.tensor([[3.0]]))


@pytest.mark.parametrize("fusion", [mixture_of_experts, mean_of_experts, wasserstein_barycenter])
def test_averaging_excludes_absent_nan_values_and_gradients(fusion):
    mean = torch.tensor([[[2.0]], [[float("nan")]]], requires_grad=True)
    log_var = torch.tensor([[[0.0]], [[float("nan")]]], requires_grad=True)
    mask = torch.tensor([[True], [False]])

    joint_mean, joint_log_var = fusion(mean, log_var, mask)
    (joint_mean + joint_log_var).sum().backward()

    torch.testing.assert_close(joint_mean, torch.tensor([[2.0]]))
    torch.testing.assert_close(joint_log_var, torch.zeros(1, 1))
    assert torch.equal(mean.grad[1], torch.zeros_like(mean.grad[1]))
    assert torch.equal(log_var.grad[1], torch.zeros_like(log_var.grad[1]))


@pytest.mark.parametrize("fusion", [mixture_of_experts, wasserstein_barycenter])
def test_averaging_zero_effective_weight_falls_back_to_equal_share(fusion):
    mean = torch.tensor([[[2.0]], [[6.0]]])
    log_var = torch.zeros_like(mean)
    mask = torch.zeros(2, 1, dtype=torch.bool)

    for weights in (None, torch.zeros(2)):
        joint_mean, joint_log_var = fusion(mean, log_var, mask, weights)
        torch.testing.assert_close(joint_mean, torch.tensor([[4.0]]))
        torch.testing.assert_close(joint_log_var, torch.zeros(1, 1))


def test_mixture_accepts_expert_weights_and_rejects_invalid_weights():
    mean = torch.tensor([[[1.0]], [[5.0]]])
    log_var = torch.zeros_like(mean)
    torch.testing.assert_close(
        mixture_of_experts(mean, log_var, weights=torch.tensor([1.0, 3.0]))[0], torch.tensor([[4.0]])
    )

    for weights in (torch.ones(3), torch.tensor([-1.0, 1.0]), torch.tensor([float("nan"), 1.0])):
        with pytest.raises(ValueError, match="weights"):
            mixture_of_experts(mean, log_var, weights=weights)

    with pytest.raises(ValueError, match="weights"):
        mixture_of_experts(mean.expand(-1, 2, -1), log_var.expand(-1, 2, -1), weights=torch.ones(2, 1))


def test_barycenter_averages_standard_deviations():
    mean = torch.tensor([[[1.0]], [[5.0]]])
    log_var = torch.tensor([[[0.0]], [[math.log(9.0)]]])

    joint_mean, joint_log_var = wasserstein_barycenter(mean, log_var)

    torch.testing.assert_close(joint_mean, torch.tensor([[3.0]]))
    assert torch.exp(joint_log_var).item() == pytest.approx(4.0, abs=1e-5)


def test_expert_groups_keep_every_expert_and_preserve_order():
    groups = expert_groups(13, 4)
    assert [len(group) for group in groups] == [4, 3, 3, 3]
    assert [index for group in groups for index in group] == list(range(13))
    assert expert_groups(3, 10) == [[0], [1], [2]]


@pytest.mark.parametrize("num_experts,num_groups", [(0, 2), (3, 0)])
def test_expert_groups_reject_invalid_counts(num_experts, num_groups):
    with pytest.raises(ValueError, match="positive"):
        expert_groups(num_experts, num_groups)


@pytest.mark.parametrize(
    "num_experts,num_groups",
    [(True, 2), (3, True), (2.0, 2), (3, 2.0)],
)
def test_expert_groups_reject_non_integer_counts(num_experts, num_groups):
    with pytest.raises(TypeError, match="must be an integer"):
        expert_groups(num_experts, num_groups)


def test_hierarchical_degenerate_groupings_match_product_and_mean():
    mean = torch.tensor([[[1.0]], [[3.0]], [[5.0]]])
    log_var = torch.zeros_like(mean)

    for got, expected in zip(hierarchical_experts(mean, log_var, num_groups=1), ProductOfExperts()(mean, log_var)):
        torch.testing.assert_close(got, expected)
    for got, expected in zip(hierarchical_experts(mean, log_var, num_groups=3), mean_of_experts(mean, log_var)):
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)


def test_hierarchical_excludes_empty_groups_for_each_sample():
    mean = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[9.0], [8.0]], [[11.0], [10.0]]])
    log_var = torch.zeros_like(mean)
    mask = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)

    joint_mean, joint_log_var = hierarchical_experts(mean, log_var, mask, num_groups=2)
    first = ProductOfExperts()(mean[:2, :1], log_var[:2, :1])
    second = ProductOfExperts()(mean[2:, 1:], log_var[2:, 1:])

    torch.testing.assert_close(joint_mean[0], first[0][0], atol=1e-5, rtol=0)
    torch.testing.assert_close(joint_log_var[0], first[1][0], atol=1e-5, rtol=0)
    torch.testing.assert_close(joint_mean[1], second[0][0], atol=1e-5, rtol=0)
    torch.testing.assert_close(joint_log_var[1], second[1][0], atol=1e-5, rtol=0)


def test_hierarchical_absent_experts_have_zero_gradient():
    mean = torch.tensor([[[1.0]], [[3.0]], [[9.0]], [[11.0]]], requires_grad=True)
    log_var = torch.zeros_like(mean, requires_grad=True)
    mask = torch.tensor([[True], [False], [True], [False]])

    joint_mean, joint_log_var = hierarchical_experts(mean, log_var, mask, num_groups=2)
    (joint_mean + joint_log_var).sum().backward()

    assert torch.equal(mean.grad[~mask], torch.zeros_like(mean.grad[~mask]))
    assert torch.equal(log_var.grad[~mask], torch.zeros_like(log_var.grad[~mask]))


def test_hierarchical_all_missing_is_finite_without_implicit_prior():
    mean = torch.tensor([[[100.0]], [[-100.0]], [[50.0]], [[-50.0]]])
    log_var = torch.zeros_like(mean)

    joint_mean, joint_log_var = hierarchical_experts(mean, log_var, torch.zeros(4, 1), num_groups=2)

    torch.testing.assert_close(joint_mean, torch.zeros(1, 1))
    assert torch.isfinite(joint_log_var).all()
    assert joint_log_var.item() > 10.0


@pytest.mark.parametrize("fusion", [mixture_of_experts, mean_of_experts, wasserstein_barycenter, hierarchical_experts])
def test_gaussian_fusions_validate_shapes_and_mask(fusion):
    mean = torch.zeros(2, 1, 1)
    with pytest.raises(ValueError, match="same nonempty"):
        fusion(mean, torch.zeros(2, 1, 2))
    with pytest.raises(ValueError, match="mask"):
        fusion(mean, mean, torch.ones(1, 1))
    with pytest.raises(ValueError, match="zero or one"):
        fusion(mean, mean, torch.tensor([[0.5], [1.0]]))


@pytest.mark.parametrize("fusion", [mixture_of_experts, mean_of_experts, wasserstein_barycenter, hierarchical_experts])
def test_gaussian_fusions_reject_invalid_tensor_types(fusion):
    mean = torch.zeros(2, 1, 1)
    with pytest.raises(TypeError, match="must be tensors"):
        fusion(mean.tolist(), mean)
    with pytest.raises(TypeError, match="same floating dtype"):
        fusion(mean, mean.double())
    with pytest.raises(TypeError, match="same floating dtype"):
        fusion(mean.long(), mean.long())
    with pytest.raises(TypeError, match="mask must be a tensor"):
        fusion(mean, mean, [[1], [1]])


@pytest.mark.parametrize("fusion", [mixture_of_experts, wasserstein_barycenter])
def test_averaging_rejects_non_tensor_weights(fusion):
    mean = torch.zeros(2, 1, 1)
    with pytest.raises(TypeError, match="weights must be a tensor"):
        fusion(mean, mean, weights=[1.0, 1.0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_averaging_low_precision_preserves_dtype(dtype):
    mean = torch.tensor([[[2.0]], [[6.0]]], dtype=dtype)
    log_var = torch.zeros_like(mean)
    for fusion in (mixture_of_experts, wasserstein_barycenter):
        joint_mean, joint_log_var = fusion(mean, log_var)
        assert joint_mean.dtype == dtype
        assert joint_log_var.dtype == dtype
        assert torch.isfinite(joint_mean).all()
        assert torch.isfinite(joint_log_var).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hierarchical_low_precision_matches_float32(dtype):
    mean = torch.tensor([[[1.0], [2.0], [5.0]], [[3.0], [4.0], [6.0]], [[9.0], [8.0], [7.0]], [[11.0], [10.0], [8.0]]])
    log_var = torch.tensor(
        [[[0.0], [1.0], [0.0]], [[-1.0], [0.5], [0.0]], [[0.5], [0.0], [0.0]], [[1.0], [-0.5], [0.0]]]
    )
    # The last sample has no present expert, so it relies on the float32 product for absent experts.
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.bool)

    joint_mean, joint_log_var = hierarchical_experts(mean.to(dtype), log_var.to(dtype), mask, num_groups=2)
    expected_mean, expected_log_var = hierarchical_experts(mean, log_var, mask, num_groups=2)

    assert joint_mean.dtype == dtype
    assert joint_log_var.dtype == dtype
    torch.testing.assert_close(joint_mean.float(), expected_mean, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(joint_log_var.float(), expected_log_var, atol=5e-2, rtol=1e-2)

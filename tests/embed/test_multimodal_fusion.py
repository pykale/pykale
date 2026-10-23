import pytest
import torch

from kale.embed.multimodal_fusion import BimodalInteractionFusion, Concat, LowRankTensorFusion, ProductOfExperts


def test_concat():
    concat = Concat()
    m1 = torch.randn(4, 3, 5)
    m2 = torch.randn(4, 2, 5)
    output = concat([m1, m2])
    assert output.shape == (4, 5 * (3 + 2))


@pytest.mark.parametrize(
    "output_type, expected_shape",
    [
        ("vector", (4, 20)),
        ("matrix", (4, 30)),
        ("scalar", (4, 20)),
    ],
)
def test_bimodal_interaction_fusion(output_type: str, expected_shape: tuple):
    mi2m_vector = BimodalInteractionFusion(input_dims=(10, 20), output_dim=30, output=output_type)
    m1 = torch.randn(4, 10)
    m2 = torch.randn(4, 20)
    output = mi2m_vector([m1, m2])
    assert output.shape == expected_shape


def test_low_rank_tensor_fusion():
    lrtf = LowRankTensorFusion(input_dims=(3, 3, 3), output_dim=1, rank=2)
    m1 = torch.randn(4, 3)
    m2 = torch.randn(4, 3)
    m3 = torch.randn(4, 3)
    output = lrtf([m1, m2, m3])
    assert output.shape == (4, 1)


def test_product_of_experts_forward():
    # Prepare dummy expert means and log_vars for 3 experts, 5 batch, 8 latent dims
    num_experts = 3
    batch_size = 5
    latent_dim = 8
    mean = torch.randn(num_experts, batch_size, latent_dim)
    log_var = torch.randn(num_experts, batch_size, latent_dim)

    poe = ProductOfExperts()
    pd_mu, pd_log_var = poe(mean, log_var)

    # Check shapes
    assert pd_mu.shape == (batch_size, latent_dim)
    assert pd_log_var.shape == (batch_size, latent_dim)
    assert isinstance(pd_mu, torch.Tensor)
    assert isinstance(pd_log_var, torch.Tensor)


def test_product_of_experts_numerical_stability():
    # All experts have large negative log_var (very low variance)
    mean = torch.zeros(2, 1, 4)
    log_var = torch.full((2, 1, 4), -20.0)
    poe = ProductOfExperts()
    pd_mu, pd_log_var = poe(mean, log_var, eps=1e-10)  # Use custom eps for coverage
    assert torch.isfinite(pd_mu).all()
    assert torch.isfinite(pd_log_var).all()


def test_product_of_experts_all_present_mask_preserves_existing_call():
    mean = torch.tensor([[[1.0, -2.0], [3.0, 4.0]], [[-1.0, 2.0], [5.0, -6.0]]], dtype=torch.float64)
    log_var = torch.tensor([[[0.2, -0.4], [0.1, 0.3]], [[-0.3, 0.5], [-0.2, 0.6]]], dtype=torch.float64)
    poe = ProductOfExperts()

    expected = poe(mean, log_var, 1e-10)
    actual = poe(mean, log_var, 1e-10, mask=torch.ones(mean.shape[:2], dtype=torch.bool))

    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        assert got.dtype == mean.dtype


def test_product_of_experts_mask_can_differ_by_subject():
    mean = torch.tensor([[[1.0], [9.0]], [[5.0], [3.0]], [[7.0], [11.0]]])
    log_var = torch.zeros_like(mean)
    mask = torch.tensor([[1, 0], [0, 1], [1, 0]])
    poe = ProductOfExperts()

    joint_mean, joint_log_var = poe(mean, log_var, mask=mask)
    first_mean, first_log_var = poe(mean[[0, 2], :1], log_var[[0, 2], :1])
    second_mean, second_log_var = poe(mean[1:2, 1:], log_var[1:2, 1:])

    torch.testing.assert_close(joint_mean[0], first_mean[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(joint_log_var[0], first_log_var[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(joint_mean[1], second_mean[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(joint_log_var[1], second_log_var[0], atol=1e-6, rtol=0)


def test_product_of_experts_absent_experts_have_no_gradient():
    mean = torch.tensor([[[1.0], [3.0]], [[4.0], [7.0]]], requires_grad=True)
    log_var = torch.zeros_like(mean, requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])

    joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=mask)
    (joint_mean + joint_log_var).sum().backward()

    assert torch.equal(mean.grad[~mask], torch.zeros_like(mean.grad[~mask]))
    assert torch.equal(log_var.grad[~mask], torch.zeros_like(log_var.grad[~mask]))
    assert torch.isfinite(mean.grad[mask]).all()
    assert torch.isfinite(log_var.grad[mask]).all()


def test_product_of_experts_all_missing_is_finite_without_implicit_prior():
    mean = torch.tensor([[[1000.0]], [[-1000.0]]])
    log_var = torch.tensor([[[-100.0]], [[100.0]]])
    joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=torch.zeros(2, 1))

    torch.testing.assert_close(joint_mean, torch.zeros(1, 1))
    assert torch.isfinite(joint_log_var).all()
    assert joint_log_var.item() > 10.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_product_of_experts_all_missing_is_finite_in_low_precision(dtype):
    mean = torch.zeros(2, 1, 1, dtype=dtype)
    log_var = torch.zeros_like(mean)
    joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=torch.zeros(2, 1))

    assert joint_mean.dtype == dtype
    assert joint_log_var.dtype == dtype
    assert torch.isfinite(joint_mean).all()
    assert torch.isfinite(joint_log_var).all()
    assert joint_log_var.item() > 10.0


def test_product_of_experts_explicit_prior_handles_missing_modalities():
    mean = torch.tensor([[[0.0]], [[1000.0]]])
    log_var = torch.tensor([[[0.0]], [[-100.0]]])
    joint_mean, joint_log_var = ProductOfExperts()(mean, log_var, mask=torch.tensor([[1], [0]]))

    torch.testing.assert_close(joint_mean, torch.zeros(1, 1))
    torch.testing.assert_close(joint_log_var, torch.zeros(1, 1), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mask", [torch.ones(1, 2), torch.tensor([[0.5, 1.0], [1.0, 0.0]])])
def test_product_of_experts_rejects_invalid_masks(mask):
    mean = torch.zeros(2, 2, 1)
    with pytest.raises(ValueError, match="mask"):
        ProductOfExperts()(mean, mean, mask=mask)


@pytest.mark.parametrize("shape", [(2, 2), (0, 2, 1)])
def test_product_of_experts_rejects_invalid_expert_shapes(shape):
    mean = torch.zeros(shape)
    with pytest.raises(ValueError, match="nonempty"):
        ProductOfExperts()(mean, mean)

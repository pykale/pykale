import pytest
import torch

from kale.embed.feature_fusion import BimodalInteractionFusion, Concat, LowRankTensorFusion, ProductOfExperts


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

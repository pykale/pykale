import pytest
import torch

from kale.embed.feature_fusion import BimodalInteractionFusion, Concat, LowRankTensorFusion


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

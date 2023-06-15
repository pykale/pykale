import pytest
import torch

from kale.predict.decode import LinearClassifier, MLPDecoder, VCDN
from kale.utils.seed import set_seed


def test_mlp_decoder():
    in_dim, hidden_dim, out_dim = 8, 16, 32
    mlp_decoder = MLPDecoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    assert mlp_decoder.fc1.weight.size() == (hidden_dim, in_dim)
    assert mlp_decoder.fc3.weight.size() == (out_dim, hidden_dim)
    assert mlp_decoder.fc4.weight.size() == (1, out_dim)
    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.size() == (16, 1)


def test_linear_classifier_model_shape():
    in_dim = 10
    out_dim = 5
    model = LinearClassifier(in_dim, out_dim)
    assert model.fc.weight.size() == (out_dim, in_dim)


def test_linear_classifier_output_shape():
    in_dim = 10
    out_dim = 5
    batch_size = 16
    x = torch.randn(batch_size, in_dim)
    model = LinearClassifier(in_dim, out_dim)
    y = model(x)
    assert y.shape == (batch_size, out_dim)


def test_linear_classifier_no_bias():
    in_dim = 10
    out_dim = 5
    model = LinearClassifier(in_dim, out_dim, bias=False)
    assert model.fc.bias is None


def test_linear_classifier_parameter_initialization():
    in_dim = 10
    out_dim = 5
    set_seed(2021)
    model = LinearClassifier(in_dim, out_dim)
    for name, param in model.named_parameters():
        if "bias" in name:
            assert torch.allclose(param.data, torch.zeros_like(param.data))
        else:
            assert param.std().detach().numpy() == pytest.approx(1 / in_dim ** 0.5, rel=1e-1)


@pytest.fixture
def vcdn():
    num_modalities = 3
    num_classes = 4
    hidden_dim = pow(num_classes, num_modalities)
    return VCDN(num_modalities=num_modalities, num_classes=num_classes, hidden_dim=hidden_dim)


def test_vcdn_forward(vcdn):
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 4)
    x3 = torch.randn(2, 4)
    output = vcdn([x1, x2, x3])
    assert output.shape == (2, 4)


def test_vcdn_reset_parameters(vcdn):
    # Set parameters to a fixed value
    with torch.no_grad():
        for param in vcdn.parameters():
            param.fill_(1.0)
    # Reset the parameters
    vcdn.reset_parameters()
    # Check that the parameters are now different
    for param in vcdn.parameters():
        assert torch.any(param != 1.0)
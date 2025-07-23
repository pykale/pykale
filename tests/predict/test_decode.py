import pytest
import torch
import torch.nn as nn

from kale.predict.decode import (
    ImageVAEDecoder,
    LinearClassifier,
    MLPDecoder,
    SignalImageFineTuningClassifier,
    SignalVAEDecoder,
    VCDN,
)
from kale.utils.seed import set_seed


def test_mlp_decoder():
    # Test with additional layers
    in_dim, hidden_dim, out_dim = 8, 16, 32
    include_decoder_layers = True
    dropout_rate = 0.1
    mlp_decoder = MLPDecoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        binary=1,
        dropout_rate=dropout_rate,
        use_deep_layers=include_decoder_layers,
        use_batchnorm=True,
    )

    layer_types = [type(layer) for layer in mlp_decoder.model]
    assert layer_types.count(nn.Linear) == 4  # 3 hidden + 1 final
    assert isinstance(mlp_decoder.model[-1], nn.Linear)
    assert mlp_decoder.model[-1].out_features == 1  # Binary classification

    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.size() == (16, 1)

    # Test without additional layers
    in_dim, hidden_dim, out_dim = 8, 16, 2
    include_decoder_layers = False
    mlp_decoder = MLPDecoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        binary=2,
        dropout_rate=dropout_rate,
        use_deep_layers=include_decoder_layers,
    )

    # Check structure
    layer_types = [type(layer) for layer in mlp_decoder.model]
    assert layer_types.count(nn.Linear) == 3  # 2 hidden + 1 final
    assert isinstance(mlp_decoder.model[-1], nn.Linear)
    assert mlp_decoder.model[-1].out_features == out_dim

    # Run forward
    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.shape == (16, out_dim)


def test_linear_classifier_shape():
    in_dim = 10
    out_dim = 5
    batch_size = 16
    x = torch.randn(batch_size, in_dim)
    model = LinearClassifier(in_dim, out_dim)
    y = model(x)
    # model shape test
    assert model.fc.weight.size() == (out_dim, in_dim)
    # output shape test
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
            assert param.std().detach().numpy() == pytest.approx(1 / in_dim**0.5, rel=1e-1)


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


def test_image_vae_decoder_forward():
    latent_dim = 8
    output_channels = 2
    batch_size = 3

    decoder = ImageVAEDecoder(latent_dim=latent_dim, output_channels=output_channels)
    decoder.eval()

    latent_vector = torch.randn(batch_size, latent_dim)
    image_recon = decoder(latent_vector)

    # Default image size is (batch_size, output_channels, 224, 224)
    assert image_recon.shape == (batch_size, output_channels, 224, 224)


def test_image_vae_decoder_repr_and_init():
    decoder = ImageVAEDecoder(latent_dim=16, output_channels=3)
    assert isinstance(decoder, ImageVAEDecoder)
    assert "ImageVAEDecoder" in repr(decoder)


def test_signal_vae_decoder_forward():
    latent_dim = 8
    output_dim = 32  # Small for test
    batch_size = 4

    decoder = SignalVAEDecoder(latent_dim=latent_dim, output_dim=output_dim)
    decoder.eval()

    latent_vector = torch.randn(batch_size, latent_dim)
    signal_recon = decoder(latent_vector)

    # The output should have shape (batch_size, 1, output_dim)
    assert signal_recon.shape == (batch_size, 1, output_dim)
    assert signal_recon.dtype == torch.float32


def test_signal_vae_decoder_init_repr():
    decoder = SignalVAEDecoder(latent_dim=16, output_dim=64)
    assert isinstance(decoder, SignalVAEDecoder)
    assert "SignalVAEDecoder" in repr(decoder)


def test_signal_vae_decoder_edge_case():
    decoder = SignalVAEDecoder(latent_dim=1, output_dim=8)
    latent_vector = torch.randn(1, 1)
    signal_recon = decoder(latent_vector)
    assert signal_recon.shape == (1, 1, 8)


# Dummy encoder that returns (mean, log_var)
class DummyEncoder(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        batch_size = x.shape[0]
        mean = torch.ones(batch_size, self.latent_dim)
        log_var = torch.zeros(batch_size, self.latent_dim)
        return mean, log_var


class DummyPretrainedModel(nn.Module):
    def __init__(self, n_latents):
        super().__init__()
        self.image_encoder = DummyEncoder(10, n_latents)
        self.signal_encoder = DummyEncoder(20, n_latents)
        self.n_latents = n_latents


def test_multimodal_classifier_forward():
    n_latents = 8
    num_classes = 3
    batch_size = 5
    hidden_dim = 16  # test the flexibility of hidden_dim

    pretrained_model = DummyPretrainedModel(n_latents=n_latents)
    classifier = SignalImageFineTuningClassifier(pretrained_model, num_classes=num_classes, hidden_dim=hidden_dim)

    # Check encoders are frozen
    for p in classifier.image_encoder.parameters():
        assert not p.requires_grad
    for p in classifier.signal_encoder.parameters():
        assert not p.requires_grad

    image = torch.randn(batch_size, 10)
    signal = torch.randn(batch_size, 20)
    logits = classifier(image, signal)
    assert logits.shape == (batch_size, num_classes)
    assert logits.requires_grad
    labels = torch.randint(0, num_classes, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()


def test_mlp_decoder_forward():
    in_dim = 64
    hidden_dim = 32
    out_dim = 16
    batch_size = 64

    # Initialize MLPDecoder model
    model = MLPDecoder(in_dim, hidden_dim, out_dim)

    # Create a mock input
    input_data = torch.randn(batch_size, in_dim)

    # Forward pass through the model
    output = model(input_data)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[0] == batch_size, "Output batch size should match input batch size"
    assert output.shape[1] == 1, "Output should have one dimension for binary classification"


def test_mlp_decoder_minimal_inputs():
    # MLPDecoder
    model = MLPDecoder(1, 1, 1)
    model.eval()
    inp = torch.randn(2, 1)
    out = model(inp)
    assert out.shape[0] == 2

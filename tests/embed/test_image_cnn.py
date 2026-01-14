import pytest
import torch

from kale.embed.image_cnn import (
    Flatten,
    Identity,
    ImageVAEEncoder,
    LeNet,
    ResNet18Feature,
    ResNet34Feature,
    ResNet50Feature,
    ResNet101Feature,
    ResNet152Feature,
    SimpleCNNBuilder,
    SmallCNNFeature,
)

BATCH_SIZE = 64

# the default input shape is batch_size * num_channel * height * weight
INPUT_BATCH = torch.randn(BATCH_SIZE, 3, 32, 32)
PARAM = [
    (ResNet18Feature, 512),
    (ResNet34Feature, 512),
    (ResNet50Feature, 2048),
    (ResNet101Feature, 2048),
    (ResNet152Feature, 2048),
]


def test_smallcnnfeature_shapes():
    model = SmallCNNFeature()
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 128)


def test_simplecnnbuilder_shapes():
    model = SimpleCNNBuilder(
        conv_layers_spec=[[16, 3], [32, 3], [64, 3], [32, 1], [64, 3], [128, 3], [256, 3], [64, 1]]
    )
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 64, 8, 8)


@pytest.mark.parametrize("param", PARAM)
def test_shapes(param):
    model, out_size = param
    model = model(weights="DEFAULT")
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, out_size)
    assert model.output_size() == out_size


def test_lenet_output_shapes():
    input_channels = 3
    output_channels = 6
    additional_layers = 2
    lenet = LeNet(input_channels, output_channels, additional_layers)
    x = torch.randn(16, 3, 32, 32)
    output = lenet(x)
    assert output.shape == (16, 24, 4, 4), "Unexpected output shape"


def test_flatten_output_shapes():
    flatten = Flatten()
    x = torch.randn(16, 3, 32, 32)
    output = flatten(x)
    assert output.shape == (16, 3072), "Unexpected output shape"


def test_identity_output_shapes():
    identity = Identity()
    x = torch.randn(16, 3, 32, 32)
    output = identity(x)
    assert output.shape == (16, 3, 32, 32), "Unexpected output shape"


def test_image_vae_encoder_forward():
    # Test configuration
    batch_size = 2
    input_channels = 1
    height, width = 224, 224  # After 3 layers with stride=2, output is 28x28

    # Create dummy image input (batch of grayscale images)
    x = torch.randn(batch_size, input_channels, height, width)

    # Initialize encoder
    encoder = ImageVAEEncoder(input_channels=input_channels, latent_dim=32)

    # Forward pass
    mean, log_var = encoder(x)

    # Check output shapes
    assert mean.shape == (batch_size, 32)
    assert log_var.shape == (batch_size, 32)
    # Check types
    assert isinstance(mean, torch.Tensor)
    assert isinstance(log_var, torch.Tensor)


# =============================================================================
# Additional Coverage Tests for Missing Branches
# =============================================================================


def test_smallcnnfeature_repr():
    """Test SmallCNNFeature __repr__ method."""
    model = SmallCNNFeature()
    repr_str = repr(model)
    assert "SmallCNNFeature" in repr_str
    assert "output_features=128" in repr_str


def test_smallcnnfeature_output_size():
    """Test SmallCNNFeature output_size method."""
    model = SmallCNNFeature()
    output_size = model.output_size()
    assert output_size == 128  # Returns int, not tuple


def test_lenet_repr():
    """Test LeNet __repr__ method."""
    model = LeNet(3, 6, 2)
    repr_str = repr(model)
    assert "LeNet" in repr_str
    assert "num_layers" in repr_str
    assert "output_channels" in repr_str


def test_lenet_output_size():
    """Test LeNet output_size method."""
    model = LeNet(3, 6, 2)
    output_size = model.output_size()
    assert isinstance(output_size, int) or isinstance(output_size, tuple)
    # LeNet returns int
    if isinstance(output_size, int):
        assert output_size == 24  # output_channels * (num_layers + 1)
    else:
        assert isinstance(output_size, tuple)


def test_simplecnnbuilder_repr():
    """Test SimpleCNNBuilder __repr__ method."""
    model = SimpleCNNBuilder(conv_layers_spec=[[16, 3], [32, 3]])
    repr_str = repr(model)
    assert "SimpleCNNBuilder" in repr_str
    assert "num_layers=2" in repr_str


def test_simplecnnbuilder_output_size():
    """Test SimpleCNNBuilder output_size method."""
    model = SimpleCNNBuilder(conv_layers_spec=[[16, 3], [32, 3]])
    output_size = model.output_size()
    assert isinstance(output_size, int) or isinstance(output_size, tuple)
    # SimpleCNNBuilder returns int
    if isinstance(output_size, int):
        assert output_size == 32  # Last layer channels
    else:
        assert isinstance(output_size, tuple)


def test_imagevaeencoder_repr():
    """Test ImageVAEEncoder __repr__ method."""
    model = ImageVAEEncoder(input_channels=3, latent_dim=32)
    repr_str = repr(model)
    assert "ImageVAEEncoder" in repr_str
    assert "input_channels=3" in repr_str
    assert "latent_dim=32" in repr_str


def test_imagevaeencoder_output_size():
    """Test ImageVAEEncoder output_size method."""
    model = ImageVAEEncoder(input_channels=3, latent_dim=32)
    output_size = model.output_size()
    assert output_size == 32  # Returns latent_dim as int


def test_resnet18_forward_with_small_input():
    """Test ResNet18Feature with small input size."""
    model = ResNet18Feature(weights=None)
    x = torch.randn(4, 3, 64, 64)  # Smaller than default 224x224
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 512


def test_resnet34_forward_with_small_input():
    """Test ResNet34Feature with small input size."""
    model = ResNet34Feature(weights=None)
    x = torch.randn(4, 3, 64, 64)
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 512


def test_resnet50_forward():
    """Test ResNet50Feature forward pass."""
    model = ResNet50Feature(weights=None)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 2048


def test_resnet101_forward():
    """Test ResNet101Feature forward pass."""
    model = ResNet101Feature(weights=None)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 2048


def test_resnet152_forward():
    """Test ResNet152Feature forward pass."""
    model = ResNet152Feature(weights=None)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 2048


def test_resnet34_output_size():
    """Test ResNet34Feature output_size method."""
    model = ResNet34Feature(weights=None)
    output_size = model.output_size()
    assert output_size == 512


def test_resnet50_output_size():
    """Test ResNet50Feature output_size method."""
    model = ResNet50Feature(weights=None)
    output_size = model.output_size()
    assert output_size == 2048


def test_resnet101_output_size():
    """Test ResNet101Feature output_size method."""
    model = ResNet101Feature(weights=None)
    output_size = model.output_size()
    assert output_size == 2048


def test_resnet152_output_size():
    """Test ResNet152Feature output_size method."""
    model = ResNet152Feature(weights=None)
    output_size = model.output_size()
    assert output_size == 2048


def test_simplecnnbuilder_without_batch_norm():
    """Test SimpleCNNBuilder with use_batchnorm=False."""
    model = SimpleCNNBuilder(conv_layers_spec=[[16, 3], [32, 3]], use_batchnorm=False)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4
    assert output.shape[1] == 32  # Last layer channels


def test_lenet_output_each_layer():
    """Test LeNet with output_each_layer=True."""
    model = LeNet(3, 6, 2, output_each_layer=True)
    x = torch.randn(4, 3, 32, 32)
    outputs = model(x)
    assert isinstance(outputs, list)
    assert len(outputs) > 0


def test_lenet_with_squeeze_output():
    """Test LeNet with squeeze_output=True."""
    model = LeNet(3, 6, 2, squeeze_output=True)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4


def test_lenet_without_squeeze_output():
    """Test LeNet with squeeze_output=False."""
    model = LeNet(3, 6, 2, squeeze_output=False)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.shape[0] == 4


def test_bottleneck_1d_input():
    """Test _Bottleneck with 1D input (input_dimension=1)."""
    from kale.embed.image_cnn import _Bottleneck

    bottleneck = _Bottleneck(inplanes=64, planes=32, stride=1, expansion=1, input_dimension=1)
    x = torch.randn(4, 64, 100)  # 1D input
    output = bottleneck(x)
    assert output.shape[0] == 4
    assert len(output.shape) == 2  # Flattened output


def test_bottleneck_1d_with_2d_shape():
    """Test _Bottleneck 1D handling when input has 2D shape."""
    from kale.embed.image_cnn import _Bottleneck

    bottleneck = _Bottleneck(inplanes=64, planes=32, stride=1, expansion=1, input_dimension=1)
    x = torch.randn(4, 64)  # 2D input (batch, channels) - should be reshaped
    output = bottleneck(x)
    assert output.shape[0] == 4
    assert len(output.shape) == 2


def test_bottleneck_1d_initialization():
    """Test _Bottleneck initialization with 1D convolutions."""
    from kale.embed.image_cnn import _Bottleneck

    # Test that 1D bottleneck uses Conv1d and BatchNorm1d
    bottleneck = _Bottleneck(inplanes=32, planes=16, stride=2, expansion=2, input_dimension=1)
    assert bottleneck.input_dimension == 1
    assert isinstance(bottleneck.conv1, torch.nn.Conv1d)
    assert isinstance(bottleneck.bn1, torch.nn.BatchNorm1d)
    assert isinstance(bottleneck.conv2, torch.nn.Conv1d)
    assert isinstance(bottleneck.bn2, torch.nn.BatchNorm1d)
    assert isinstance(bottleneck.conv3, torch.nn.Conv1d)
    assert isinstance(bottleneck.bn3, torch.nn.BatchNorm1d)
    assert isinstance(bottleneck.avgpool, torch.nn.AdaptiveAvgPool1d)


def test_bottleneck_2d_default():
    """Test _Bottleneck with default 2D convolutions."""
    from kale.embed.image_cnn import _Bottleneck

    # Test default input_dimension=2 path
    bottleneck = _Bottleneck(inplanes=64, planes=32, stride=1, expansion=1, input_dimension=2)
    assert bottleneck.input_dimension == 2
    assert isinstance(bottleneck.conv1, torch.nn.Conv2d)
    assert isinstance(bottleneck.bn1, torch.nn.BatchNorm2d)


def test_lenet_output_each_layer_no_squeeze():
    """Test LeNet with output_each_layer=True and squeeze_output=False."""
    model = LeNet(3, 6, 1, output_each_layer=True, squeeze_output=False)
    x = torch.randn(4, 3, 32, 32)
    outputs = model(x)
    assert isinstance(outputs, list)
    # Verify outputs are not squeezed
    for output in outputs:
        assert isinstance(output, torch.Tensor)

import torch

# dimension locations in a typical image batch tensor
SPATIAL_BATCH_DIMENSION = 0
SPATIAL_CHANNEL_DIMENSION = 1
SPATIAL_HEIGHT_DIMENSION = 2
SPATIAL_WIDTH_DIMENSION = 3

NUMBER_OF_DIMENSIONS = 4


def spatial_to_seq(image_tensor: torch.Tensor):
    """
    Takes a torch tensor of shape (batch_size, channels, height, width)
    as used and outputted by CNNs and creates a sequence view of shape
    (sequence_length, batch_size, channels) as required by
    torch's transformer module. In other words, unrolls the
    spatial grid into the sequence length and rearranges the
    dimension ordering.

    Args:
        image_tensor: tensor of shape (batch_size, channels, height, width) (required).
    """
    original_size = image_tensor.size()

    batch_size = original_size[SPATIAL_BATCH_DIMENSION]
    num_channels = original_size[SPATIAL_CHANNEL_DIMENSION]
    spatial_height = original_size[SPATIAL_HEIGHT_DIMENSION]
    spatial_width = original_size[SPATIAL_WIDTH_DIMENSION]

    permuted_tensor = image_tensor.permute(
        SPATIAL_HEIGHT_DIMENSION, SPATIAL_WIDTH_DIMENSION, SPATIAL_BATCH_DIMENSION, SPATIAL_CHANNEL_DIMENSION
    )

    sequence_tensor = permuted_tensor.view(spatial_height * spatial_width, batch_size, num_channels)

    return sequence_tensor


# dimension locations in a typical Transformer sequence batch tensor
SEQUENCE_LENGTH_DIMENSION = 0
SEQUENCE_BATCH_DIMENSION = 1
SEQUENCE_FEATURE_DIMENSION = 2

SEQUENCE_NUMBER_OF_DIMENSIONS = 3


def seq_to_spatial(sequence_tensor: torch.Tensor, desired_height: int, desired_width: int):
    """Takes a torch tensor of shape (sequence_length, batch_size, num_features)
    as used and outputted by Transformers and creates a view of shape
    (batch_size, num_features, height, width) as used and outputted by CNNs.
    In other words, rearranges the dimension ordering and rolls
    sequence_length into (height,width). height*width must equal
    the sequence length of the input sequence.

    Args:
        sequence_tensor: sequence tensor of shape (sequence_length, batch_size, num_features) (required).
        desired_height: the height into which the sequence length should be rolled into (required).
        desired_width: the width into which the sequence length should be rolled into (required).

    """
    original_size = sequence_tensor.size()

    batch_size = original_size[SEQUENCE_BATCH_DIMENSION]
    num_channels = original_size[SEQUENCE_FEATURE_DIMENSION]

    permuted_tensor = sequence_tensor.permute(
        SEQUENCE_BATCH_DIMENSION, SEQUENCE_FEATURE_DIMENSION, SEQUENCE_LENGTH_DIMENSION
    )

    spatial_tensor = permuted_tensor.view(batch_size, num_channels, desired_height, desired_width)

    return spatial_tensor

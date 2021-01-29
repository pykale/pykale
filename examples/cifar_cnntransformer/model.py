"""
Define and build the model based on chosen hyperparameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.embed.attention_cnn import CNNTransformer, ContextCNNGeneric


class SimpleCNN(nn.Module):
    """
    A builder for simple CNNs to experiment with different
    basic architectures as specified in config.py.
    """

    activations = {"relu": nn.ReLU(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU()}

    def __init__(self, conv_layers_spec, activation_fun, use_batchnorm, pool_locations):
        """
        Parameter meanings explained in the config file.
        """
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        activation_fun = self.activations[activation_fun]

        # Repetitively adds a convolution, batchnorm, activationFunction,
        # and maxpooling layer.
        for layer_num, (num_kernels, kernel_size) in enumerate(conv_layers_spec):
            conv = nn.Conv2d(in_channels, num_kernels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.layers.append(conv)

            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(num_kernels))

            self.layers.append(activation_fun)

            if layer_num in pool_locations:
                self.layers.append(nn.MaxPool2d(kernel_size=2))

            in_channels = num_kernels

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x


class PredictionHead(nn.Module):
    """
    Simple classification prediction-head block to plug ontop of the 4D
    output of a CNN.
    Args:
        num_classes: the number of different classes that can be predicted.
        input_shape: the shape that input to this head will have. Expected
                      to be (batch_size, channels, height, width)
    """

    def __init__(self, num_classes, input_shape):
        super(PredictionHead, self).__init__()
        self.avgpool = nn.AvgPool2d(input_shape[2])
        self.linear = nn.Linear(input_shape[1], num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return F.log_softmax(x, 1)


def get_model(cfg):
    """
    Builds and returns a model according to the config
    object passed.

    Args:
        cfg: A YACS config object.
    """

    cnn = SimpleCNN(cfg.CNN.CONV_LAYERS, cfg.CNN.ACTIVATION_FUN, cfg.CNN.USE_BATCHNORM, cfg.CNN.POOL_LOCATIONS)

    if cfg.TRANSFORMER.USE_TRANSFORMER:
        context_cnn = CNNTransformer(
            cnn,
            cfg.CNN.OUTPUT_SHAPE,
            cfg.TRANSFORMER.NUM_LAYERS,
            cfg.TRANSFORMER.NUM_HEADS,
            cfg.TRANSFORMER.DIM_FEEDFORWARD,
            cfg.TRANSFORMER.DROPOUT,
            cfg.TRANSFORMER.OUTPUT_TYPE,
        )
    else:
        context_cnn = ContextCNNGeneric(
            cnn, cfg.CNN.OUTPUT_SHAPE, contextualizer=lambda x: x, output_type=cfg.TRANSFORMER.OUTPUT_TYPE
        )

    classifier = PredictionHead(cfg.DATASET.NUM_CLASSES, cfg.CNN.OUTPUT_SHAPE)
    return nn.Sequential(context_cnn, classifier)

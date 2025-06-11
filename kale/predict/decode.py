"""
Provides implementations of various decoders based on neural network modules for prediction and classification tasks.
Refer to the PyTorch documentation for the accompanying tutorial on neural network modules:
https://pytorch.org/docs/stable/generated/torch.nn.Module.html
"""

from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.embed.gripnet import GripNet
from kale.evaluate.metrics import auprc_auroc_ap
from kale.prepdata.graph_negative_sampling import typed_negative_sampling
from kale.prepdata.supergraph_construct import SuperGraph
from kale.utils.initialize_nn import bias_init, xavier_init


class MLPDecoder(nn.Module):
    """
    A generalized MLP model that can act as either a 2-layer MLPDecoder or a 4-layer MLPDecoder based on the include_decoder_layers parameter.

    Args:
        in_dim (int): the dimension of input feature.
        hidden_dim (int): the dimension of hidden layers.
        out_dim (int): the dimension of output layer.
        dropout_rate (float): the dropout rate during training.
        include_decoder_layers (bool): whether or not to include the additional layers that are part of the MLPDecoder
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1, include_decoder_layers=False):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.include_decoder_layers = include_decoder_layers

        if self.include_decoder_layers:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, out_dim)
            self.fc4 = nn.Linear(out_dim, 1)
            torch.nn.init.normal_(self.fc4.weight)
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.include_decoder_layers:
            x = self.dropout(F.relu(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

        return x


class DistMultDecoder(torch.nn.Module):
    """
    Build `DistMult
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf>`_ factorization
    as GripNet decoder in PoSE dataset.
    Copy-paste with slight modifications from https://github.com/NYXFLOWER/GripNet.

    Args:
        in_channels (int): the dimension of input feature.
        num_edge_type (int): the number of edge types.
    """

    def __init__(self, in_channels: int, num_edge_type: int):
        super(DistMultDecoder, self).__init__()
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(num_edge_type, in_channels))

        self.reset_parameters()

    def forward(self, x, edge_index: torch.Tensor, edge_type: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        """
        Args:
            x: the input node feature embeddings.
            edge_index: the edge index in COO format with shape [2, num_edges].
            edge_type: the one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: whether to use sigmoid function or not.
        """
        value = (x[edge_index[0]] * x[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_channels))

    def __repr__(self) -> str:
        return "{}: DistMultLayer(in_channels={}, num_relations={})".format(
            self.__class__.__name__, self.in_channels, self.num_edge_type
        )


class GripNetLinkPrediction(pl.LightningModule):
    """
    Build GripNet-DistMult (encoder-decoder) model for link prediction.

    Args:
        supergraph (SuperGraph): the input supergraph.
        learning_rate (float): the learning rate for training.
        epsilon (float, optional): a small number in loss function to improve numerical stability. Defaults to 1e-13.
    """

    def __init__(self, supergraph: SuperGraph, learning_rate: float, epsilon: float = 1e-13):
        super().__init__()

        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.encoder = GripNet(supergraph)
        self.decoder = self.__init_decoder__()

    def __init_decoder__(self) -> DistMultDecoder:
        in_channels = self.encoder.out_channels
        supergraph = self.encoder.supergraph
        task_supervertex_name = supergraph.topological_order[-1]
        num_edge_type = supergraph.supervertex_dict[task_supervertex_name].num_edge_type

        # get the number of nodes on the task-associated supervertex
        self.num_task_nodes = supergraph.supervertex_dict[task_supervertex_name].num_node

        return DistMultDecoder(in_channels, num_edge_type)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_type_range: torch.Tensor) -> Tuple:
        x = self.encoder()

        pos_score = self.decoder(x, edge_index, edge_type)
        pos_loss = -torch.log(pos_score + self.epsilon).mean()

        edge_index = typed_negative_sampling(edge_index, self.num_task_nodes, edge_type_range)

        neg_score = self.decoder(x, edge_index, edge_type)
        neg_loss = -torch.log(1 - neg_score + self.epsilon).mean()

        loss = pos_loss + neg_loss

        # compute averaged metric scores over edge types
        num_edge_type = edge_type_range.shape[0]
        record = []
        for i in range(num_edge_type):
            start, end = edge_type_range[i]
            pos_score_this_type, neg_score_this_type = pos_score[start:end], neg_score[start:end]

            score = torch.cat([pos_score_this_type, neg_score_this_type])
            target = torch.cat([torch.ones(pos_score_this_type.shape[0]), torch.zeros(neg_score_this_type.shape[0])])

            record.append(list(auprc_auroc_ap(target, score)))
        auprc, auroc, ave_precision = np.array(record).mean(axis=0)

        return loss, auprc, auroc, ave_precision

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def __step__(self, batch, mode="train"):
        edge_index, edge_type, edge_type_range = batch
        loss, auprc, auroc, ap = self.forward(
            edge_index.reshape((2, -1)), edge_type.flatten(), edge_type_range.reshape((-1, 2))
        )

        if mode == "train" or "val":
            self.log(f"{mode}_loss", loss)
        else:
            self.log(f"{mode}_auprc", auprc)
            self.log(f"{mode}_auroc", auroc)
            self.log(f"{mode}_ap@50", ap)

        return loss

    def training_step(self, batch, batch_idx):
        return self.__step__(batch)

    def validation_step(self, batch, batch_idx):
        return self.__step__(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.__step__(batch, mode="test")

    def __repr__(self) -> str:
        return "{}: \nEncoder: {} ModuleDict(\n{})\n Decoder: {}".format(
            self.__class__.__name__, self.encoder.__class__.__name__, self.encoder.supervertex_module_dict, self.decoder
        )


class LinearClassifier(nn.Module):
    r"""Build a linear transformation module.

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. (default: ``True``)
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters of the model."""
        self.fc.apply(xavier_init)
        self.fc.apply(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class VCDN(nn.Module):
    r"""The View Correlation Discovery Network (VCDN) to learn the higher-level intra-view and cross-view correlations
    in the label space, implemented according to the method described in 'MOGONET integrates multi-omics data using
    graph convolutional networks allowing patient classification and biomarker identification'
    - Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021).

    Args:
        num_modalities (int): The total number of modalities in the dataset.
        num_classes (int): The total number of classes in the dataset.
        hidden_dim (int): Size of the hidden layer.
    """

    def __init__(self, num_modalities: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()

        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(pow(self.num_classes, self.num_modalities), hidden_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters of the model."""
        self.model.apply(xavier_init)
        self.model.apply(bias_init)

    def forward(self, multimodal_input: List[torch.Tensor]) -> torch.Tensor:
        for modality in range(self.num_modalities):
            multimodal_input[modality] = torch.sigmoid(multimodal_input[modality])
        x = torch.reshape(
            torch.matmul(multimodal_input[0].unsqueeze(-1), multimodal_input[1].unsqueeze(1)),
            (-1, pow(self.num_classes, 2), 1),
        )
        for modality in range(2, self.num_modalities):
            x = torch.reshape(
                torch.matmul(x, multimodal_input[modality].unsqueeze(1)), (-1, pow(self.num_classes, modality + 1), 1)
            )
        input_tensor = torch.reshape(x, (-1, pow(self.num_classes, self.num_modalities)))
        output = self.model(input_tensor)

        return output


class ImageVaeDecoder(nn.Module):
    """
    ImageVAEDecoder reconstructs 2D image data from a latent representation in a variational autoencoder (VAE) framework.

    The decoder transforms a latent vector back into an image by first projecting it through a fully connected layer,
    then applying a sequence of transposed convolutional layers to progressively upsample and reconstruct the original
    spatial dimensions. The output activation is a sigmoid, suitable for producing pixel values in [0, 1] for normalized images.

    Args:
        latent_dim (int, optional): Dimensionality of the input latent vector. Default is 256.
        output_channels (int, optional): Number of channels in the output image (e.g., 1 for grayscale). Default is 1.

        Note:
            This decoder is specifically designed for reconstructing 224×224 resolution images. The latent-to-feature
            projection and transposed convolutions are fixed to match this output size, consistent with a typical
            encoder-decoder setup using 224×224 inputs.

    Forward Input:
        latent_vector (Tensor): Latent vector of shape (batch_size, latent_dim).

    Forward Output:
        image_recon (Tensor): Reconstructed image tensor of shape (batch_size, output_channels, H, W).

    Example:
        decoder = ImageVaeDecoder(latent_dim=128, output_channels=1)
        image_recon = decoder(latent_vector)
    """

    def __init__(self, latent_dim=256, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 28 * 28)
        self.convtrans1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, latent_vector):
        latent_vector = self.fc(latent_vector)
        latent_vector = latent_vector.view(-1, 64, 28, 28)
        latent_vector = self.relu(self.convtrans1(latent_vector))
        latent_vector = self.relu(self.convtrans2(latent_vector))
        image_recon = self.output_activation(self.convtrans3(latent_vector))
        return image_recon


class SignalVaeDecoder(nn.Module):
    """
    SignalVAEDecoder reconstructs 1D signal data from a latent representation in a variational autoencoder (VAE) framework.

    The decoder projects the latent vector through a fully connected layer and then upsamples it with a stack of 1D
    transposed convolutional layers, gradually reconstructing the original signal's temporal structure.
    The final layer is a linear activation (identity), making it suitable for reconstructing standardized or real-valued signals.

    Args:
        latent_dim (int, optional): Dimensionality of the input latent vector. Default is 256.
        output_dim (int, optional): Length of the output 1D signal. Default is 60000.

        Note:
            This decoder is designed to reconstruct signals with a fixed output length of `output_dim`, which must be divisible by 8.
            The latent vector is reshaped to a (64, output_dim // 8) feature map and passed through three transposed convolutional layers
            that double the temporal resolution at each stage, resulting in the final signal length.

            Suitable for use with preprocessed 12-lead ECG signals sampled at 500 Hz (i.e., over 2 minutes, resulting in 60,000 timepoints).

    Forward Input:
        latent_vector (Tensor): Latent vector of shape (batch_size, latent_dim).

    Forward Output:
        signal_recon (Tensor): Reconstructed 1D signal tensor of shape (batch_size, 1, output_dim).

    Example:
        decoder = SignalVaeDecoder(latent_dim=128, output_dim=60000)
        signal_recon = decoder(latent_vector)
    """

    def __init__(self, latent_dim=256, output_dim=60000):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * (output_dim // 8))
        self.convtrans1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.output_activation = nn.Identity()

    def forward(self, latent_vector):
        latent_vector = self.fc(latent_vector)
        latent_vector = latent_vector.view(-1, 64, latent_vector.size(1) // 64)
        latent_vector = self.relu(self.convtrans1(latent_vector))
        latent_vector = self.relu(self.convtrans2(latent_vector))
        signal_recon = self.output_activation(self.convtrans3(latent_vector))
        return signal_recon


class SignalImageFineTuningClassifier(nn.Module):
    """
    MultimodalClassifier performs supervised classification using frozen encoders from a pre-trained multimodal model.

    This classifier takes feature representations from both an image and a signal encoder (frozen from pre-training),
    concatenates the mean latent vectors, and passes them through a fully-connected neural network for downstream
    classification tasks. This approach leverages robust multimodal feature learning for scenarios such as medical imaging,
    sensor fusion, or any paired data where both modalities are available at inference.

    Args:
        pretrained_model (kale.embed.multimodal_encoder.SignalImageVAE): Pre-trained multimodal model.
        num_classes (int): Number of output classes.
        hidden_dim (int, optional): Number of hidden units in the classifier. Default is 128.


    Forward Input:
        image (Tensor): Batch of image modality features, shape (batch_size, channels, height, width).
        signal (Tensor): Batch of 1D signal modality features, shape (batch_size, channels, length).

    Forward Output:
        logits (Tensor): Unnormalized classification scores, shape (batch_size, num_classes).

    Example:
        model = MultimodalClassifier(pretrained_model, num_classes=3)
        logits = model(images, signals)
    """

    def __init__(self, pretrained_model, num_classes, hidden_dim=128):
        super().__init__()
        self.image_encoder = pretrained_model.image_encoder
        self.signal_encoder = pretrained_model.signal_encoder

        # Freeze the encoder weights
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.signal_encoder.parameters():
            param.requires_grad = False

        combined_feature_dim = pretrained_model.n_latents * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image, signal):
        """
        Forward pass through the multimodal classifier.

        Args:
            image (Tensor): Batch of image data.
            signal (Tensor): Batch of signal data.

        Returns:
            logits (Tensor): Classification outputs before softmax.
        """
        mu_image, _ = self.image_encoder(image)
        mu_signal, _ = self.signal_encoder(signal)
        combined_features = torch.cat((mu_image, mu_signal), dim=1)
        logits = self.classifier(combined_features)
        return logits

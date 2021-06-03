# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk or xianyuan.liu@outlook.com
# =============================================================================

"""
Define the feature extractor for video including I3D, R3D_18, MC3_18 and R2PLUS1D_18 w/o SELayers.
"""

import logging

from torch import nn

from kale.embed.video_i3d import i3d_joint
from kale.embed.video_res3d import mc3, r2plus1d, r3d
from kale.embed.video_se_i3d import se_i3d_joint
from kale.embed.video_se_res3d import se_mc3, se_r2plus1d, se_r3d
from kale.embed.video_selayer import SELayer4feat
from kale.embed.video_transformer import TransformerBlock
from kale.loaddata.video_access import get_image_modality


def get_feat_extractor4video(model_name, image_modality, attention, dict_num_classes):
    """
    Get the feature extractor w/o the pre-trained model and SELayers. The pre-trained models are saved in the path
    ``$XDG_CACHE_HOME/torch/hub/checkpoints/``. For Linux, default path is ``~/.cache/torch/hub/checkpoints/``.
    For Windows, default path is ``C:/Users/$USER_NAME/.cache/torch/hub/checkpoints/``.
    Provide four pre-trained models: "rgb_imagenet", "flow_imagenet", "rgb_charades", "flow_charades".

    Args:
        model_name (string): The name of the feature extractor. (Choices=["I3D", "R3D_18", "R2PLUS1D_18", "MC3_18"])
        image_modality (string): Image type. (Choices=["rgb", "flow", "joint"])
        attention (string): The attention type. (Choices=["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerCT", "SELayerTC", "SELayerMAC"])
        dict_num_classes (dict): The dictionary of class number for specific dataset.

    Returns:
        feature_network (dictionary): The network to extract features.
        class_feature_dim (int): The dimension of the feature network output for ClassNet.
                            It is a convention when the input dimension and the network is fixed.
        domain_feature_dim (int): The dimension of the feature network output for DomainNet.
    """
    rgb, flow, audio = get_image_modality(image_modality)
    # only use verb class when input is image.
    num_classes = dict_num_classes["verb"]

    attention_list = ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerCT", "SELayerTC", "SELayerMAC"]
    model_list = ["I3D", "R3D_18", "MC3_18", "R2PLUS1D_18"]

    if attention in attention_list:
        att = True
    elif attention == "None":
        att = False
    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current: {}".format(attention))

    if model_name not in model_list:
        raise ValueError("Wrong MODEL.METHOD. Current:{}".format(model_name))

    # Get I3D w/o SELayers for RGB, Flow or joint input
    if model_name == "I3D":
        rgb_pretrained_model = flow_pretrained_model = None
        if rgb:
            rgb_pretrained_model = "rgb_imagenet"  # Options=["rgb_imagenet", "rgb_charades"]
        if flow:
            flow_pretrained_model = "flow_imagenet"  # Options=["flow_imagenet", "flow_charades"]

        if rgb and flow:
            class_feature_dim = 2048
            domain_feature_dim = class_feature_dim / 2
        else:
            class_feature_dim = 1024
            domain_feature_dim = class_feature_dim

        if not att:
            logging.info("{} without SELayer.".format(model_name))
            feature_network = i3d_joint(
                rgb_pt=rgb_pretrained_model, flow_pt=flow_pretrained_model, num_classes=num_classes, pretrained=True
            )
        else:
            logging.info("{} with {}.".format(model_name, attention))
            feature_network = se_i3d_joint(
                rgb_pt=rgb_pretrained_model,
                flow_pt=flow_pretrained_model,
                attention=attention,
                num_classes=num_classes,
                pretrained=True,
            )

    # Get R3D_18/R2PLUS1D_18/MC3_18 w/o SELayers for RGB, Flow or joint input
    elif model_name in ["R3D_18", "R2PLUS1D_18", "MC3_18"]:
        if rgb and flow:
            class_feature_dim = 1024
            domain_feature_dim = class_feature_dim / 2
        else:
            class_feature_dim = 512
            domain_feature_dim = class_feature_dim

        if model_name == "R3D_18":
            if not att:
                logging.info("{} without SELayer.".format(model_name))
                feature_network = r3d(rgb=rgb, flow=flow, pretrained=True)
            else:
                logging.info("{} with {}.".format(model_name, attention))
                feature_network = se_r3d(rgb=rgb, flow=flow, pretrained=True, attention=attention)

        elif model_name == "R2PLUS1D_18":
            if not att:
                logging.info("{} without SELayer.".format(model_name))
                feature_network = r2plus1d(rgb=rgb, flow=flow, pretrained=True)
            else:
                logging.info("{} with {}.".format(model_name, attention))
                feature_network = se_r2plus1d(rgb=rgb, flow=flow, pretrained=True, attention=attention)

        elif model_name == "MC3_18":
            if not att:
                logging.info("{} without SELayer.".format(model_name))
                feature_network = mc3(rgb=rgb, flow=flow, pretrained=True)
            else:
                logging.info("{} with {}.".format(model_name, attention))
                feature_network = se_mc3(rgb=rgb, flow=flow, pretrained=True, attention=attention)
    feature_network.update({"audio": None})
    return feature_network, int(class_feature_dim), int(domain_feature_dim)


class BoringNetVideo(nn.Module):
    """Regular simple network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
    """

    def __init__(self, input_size=512, n_channel=512, n_out=256, dropout_keep_prob=0.5):
        super(BoringNetVideo, self).__init__()
        # self.hidden_sizes = 512
        # self.num_layers = 4

        # self.conv3d = nn.Conv3d(in_channels=input_size, out_channels=512, kernel_size=(1, 1, 1))
        # self.fc1 = nn.Linear(input_size, n_channel)
        # self.bn1 = nn.BatchNorm1d(n_channel)
        # self.relu1 = nn.ReLU()
        # self.dp1 = nn.Dropout(dropout_keep_prob)
        # self.fc2 = nn.Linear(n_channel, n_channel)
        # self.relu2 = nn.ReLU()
        # self.dp2 = nn.Dropout(dropout_keep_prob)
        # self.fc3 = nn.Linear(n_channel, n_out)
        # self.transformer = nn.ModuleList(
        #     [
        #         TransformerBlock(
        #             emb_dim=input_size,
        #             num_heads=8,
        #             att_dropout=0.1,
        #             att_resid_dropout=0.1,
        #             final_dropout=0.1,
        #             max_seq_len=9,
        #             ff_dim=self.hidden_sizes,
        #             causal=False,
        #         )
        #         for _ in range(self.num_layers)
        #     ]
        # )

        # self.transformer1 = TransformerBlock(
        #     emb_dim=input_size,
        #     num_heads=8,
        #     att_dropout=0.1,
        #     att_resid_dropout=0.1,
        #     final_dropout=0.1,
        #     max_seq_len=9,
        #     ff_dim=input_size,
        # )
        # self.transformer2 = TransformerBlock(
        #     emb_dim=input_size,
        #     num_heads=8,
        #     att_dropout=0.1,
        #     att_resid_dropout=0.1,
        #     final_dropout=0.1,
        #     max_seq_len=9,
        #     ff_dim=input_size,
        # )
        # self.transformer3 = TransformerBlock(
        #     emb_dim=input_size,
        #     num_heads=8,
        #     att_dropout=0.1,
        #     att_resid_dropout=0.1,
        #     final_dropout=0.1,
        #     max_seq_len=9,
        #     ff_dim=input_size,
        # )
        # self.transformer4 = TransformerBlock(
        #     emb_dim=input_size,
        #     num_heads=8,
        #     att_dropout=0.1,
        #     att_resid_dropout=0.1,
        #     final_dropout=0.1,
        #     max_seq_len=9,
        #     ff_dim=input_size,
        # )
        self.fc1 = nn.Linear(input_size, n_channel)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_keep_prob)
        self.fc2 = nn.Linear(n_channel, n_out)
        self.selayer1 = SELayer4feat(channel=8, reduction=2)

        # self.dim_reduction_layer = torch.nn.Identity()
        #
        # self.classification_vector = nn.Parameter(torch.randn(1, 1, input_size))
        # self.pos_encoding = nn.Parameter(
        #     torch.randn(1, 9, input_size)
        # )

    def forward(self, x):
        # x = x.squeeze()
        # x = self.fc1(x)
        # x = self.dp1(self.relu1(self.bn1(self.fc1(x))))
        # x = self.dp1(self.relu1(self.fc1(x)))
        # x = self.dp2(self.relu2(self.fc2(x)))
        # x = self.fc3(x)
        # x = self.transformer1(x)
        # x = self.transformer2(x)
        # x = self.transformer3(x)
        # x = self.transformer4(x)

        # (B, F, INPUT_DIM) -> (B, F, D)

        # x = self.dim_reduction_layer(x)
        # B, F, D = x.size()

        # classification_vector = self.classification_vector.repeat((B, 1, 1))
        # (B, F, D) -> (B, 1+F, D)
        # x = torch.cat([classification_vector, x], dim=1)
        # seq_len = x.size(1)
        # for layer in self.transformer:
        #     # x = x + self.pos_encoding[:, :seq_len, :]
        #     x = layer(x)
        x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
        x = self.selayer1(x)
        return x


def get_feat_extractor4feature(attention, image_modality, num_classes, num_out=256):
    feature_network_rgb = feature_network_flow = feature_network_audio = None
    rgb, flow, audio = get_image_modality(image_modality)
    if rgb:
        feature_network_rgb = BoringNetVideo(input_size=1024, n_out=num_out)
    if flow:
        feature_network_flow = BoringNetVideo(input_size=1024, n_out=num_out)
    if audio:
        feature_network_audio = BoringNetVideo(input_size=1024, n_out=num_out)

    domain_feature_dim = int(num_out * 8)
    if rgb:
        if flow:
            if audio:  # For all inputs
                class_feature_dim = int(domain_feature_dim * 3)
            else:  # For joint(rgb+flow) input
                class_feature_dim = int(domain_feature_dim * 2)
        else:
            if audio:  # For rgb+audio input
                class_feature_dim = int(domain_feature_dim * 2)
            else:  # For rgb input
                class_feature_dim = domain_feature_dim
    else:
        if flow:
            if audio:  # For flow+audio input
                class_feature_dim = int(domain_feature_dim * 2)
            else:  # For flow input
                class_feature_dim = domain_feature_dim
        else:  # For audio input
            class_feature_dim = domain_feature_dim

    return (
        {"rgb": feature_network_rgb, "flow": feature_network_flow, "audio": feature_network_audio},
        int(class_feature_dim),
        int(domain_feature_dim),
    )

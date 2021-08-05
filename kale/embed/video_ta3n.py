from torch import nn
from torch.nn.init import normal_, constant_
from torchvision.models.utils import load_state_dict_from_url

from kale.predict.class_domain_nets import ClassNetTA3NFrame, ClassNetTA3NVideo, DomainNetTA3N
from kale.embed.video_i3d import InceptionI3d
from kale.embed.video_selayer import SELayer4feat
from kale.embed.video_transformer import TransformerBlock
from kale.embed.video_trn import TRNRelationModuleMultiScale, TRNRelationModule

model_urls_ta3n = {
    "rgb_ta3n": None,
    "flow_ta3n": None,
    "audio_ta3n": None,
}


class TA3NSpatialBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=512, input_type="feature", add_fc=1, dropout_rate=0.5,
                 num_classes=[10]):
        super(TA3NSpatialBlock, self).__init__()
        self.add_fc = add_fc
        self.input_type = input_type
        self.std = 0.001
        self.relu = nn.ReLU(inplace=True)
        # TODO: remove _i later
        self.dp_i = nn.Dropout(p=dropout_rate)
        # self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        # self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

        if self.input_type == "image":
            # TODO: fe for image input
            self.feature_net = InceptionI3d(in_channels=3, num_classes=num_classes)
        else:
            self.fc1 = nn.Linear(input_size, output_size)
            normal_(self.fc1.weight, 0, self.std)
            constant_(self.fc1.bias, 0)

            if self.add_fc > 1:
                self.fc2 = nn.Linear(output_size, output_size)
                normal_(self.fc2.weight, 0, self.std)
                constant_(self.fc2.bias, 0)

            if self.add_fc > 2:
                self.fc3 = nn.Linear(output_size, output_size)
                normal_(self.fc3.weight, 0, self.std)
                constant_(self.fc3.bias, 0)

    def forward(self, input):
        if self.input_type == "feature":
            pass
        elif self.input_type == "image":
            x = self.feature_net(input)
        else:
            raise ValueError("Input type is not in [feature, image]. Current is {}".format(self.input_type))

        x = input.view(-1, input.size()[-1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_i(x)

        if self.add_fc > 1:
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout_i(x)
        if self.add_fc > 2:
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout_i(x)
        return x


# class TA3NSpatialVerbClassifier(nn.Module):
#     def __init__(self, input_size=512, output_size=97):
#         super(TA3NSpatialVerbClassifier, self).__init__()
#         self.std = 0.001
#         self.fc = nn.Linear(input_size, output_size)
#         normal_(self.fc.weight, 0, self.std)
#         constant_(self.fc.bias, 0)
#
#     def forward(self, input):
#         x = self.fc(input)
#         return x
#
#
# class TA3NSpatialNounClassifier(nn.Module):
#     def __init__(self, input_size=512, output_size=300):
#         super(TA3NSpatialNounClassifier, self).__init__()
#         self.std = 0.001
#         self.fc = nn.Linear(input_size, output_size)
#         normal_(self.fc.weight, 0, self.std)
#         constant_(self.fc.bias, 0)
#
#     def forward(self, input):
#         x = self.fc(input)
#         return x


# class TA3NSpatialDomainNet(nn.Module):
#     def __init__(self, input_size=512, output_size=512):
#         super(TA3NSpatialDomainNet, self).__init__()
#         self.std = 0.001
#
#         self.fc1 = nn.Linear(input_size, output_size)
#         normal_(self.fc1.weight, 0, self.std)
#         constant_(self.fc1.bias, 0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(output_size, 2)
#         normal_(self.fc2.weight, 0, self.std)
#         constant_(self.fc2.bias, 0)
#
#     def forward(self, input):
#         x = self.fc1(input)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# class TA3NVideoDomainNet(nn.Module):
#     def __init__(self, input_size=256, output_size=256):
#         super(TA3NVideoDomainNet, self).__init__()
#         self.std = 0.001
#
#         self.fc1 = nn.Linear(input_size, output_size)
#         normal_(self.fc1.weight, 0, self.std)
#         constant_(self.fc1.bias, 0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(output_size, 2)
#         normal_(self.fc2.weight, 0, self.std)
#         constant_(self.fc2.bias, 0)
#
#     def forward(self, input):
#         x = self.fc1(input)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


def ta3n(name, input_size=1024, output_size=256,
         input_type="feature", frame_aggregation="tsn-m", segments=5):
    """Get TA3N module with pretrained model."""
    spatial_model = TA3NSpatialBlock(input_size=input_size, input_type=input_type)
    if frame_aggregation == "trn":
        num_bottleneck = 512
        temporal_model = TRNRelationModule(output_size, num_bottleneck, segments)
    elif frame_aggregation == "trn-m":
        num_bottleneck = 256
        temporal_model = TRNRelationModuleMultiScale(output_size, num_bottleneck, segments)
    # TODO: Add more temporal blocks from ta3n
    return {"spatial": spatial_model, "temporal": temporal_model}


def ta3n_joint(
        rgb=False,
        flow=False,
        audio=False,
        input_size=1024,
        output_size=256,
        input_type="feature",
        frame_aggregation="tsn-m",
        segments=5,
        dict_n_class={},
):
    """Get TA3N model for different inputs.

    Args:
        rgb_name (string, optional): the name of pre-trained model for rgb input. (Default: None)
        flow_name (string, optional): the name of pre-trained model for flow input. (Default: None)
        audio_name (string, optional): the name of pre-trained model for audio input. (Default: None)
        pretrained (bool, optional): choose if pretrained parameters are used. (Default: False)
        input_size (int, optional): dimension of the final feature vector. (Defaults to 1024)
        n_out (int, optional): dimension of the output feature vector. (Defaults to 256)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)
        input_type (string): the type of input. (Choices=["image", "feature"])
        dict_n_class (int): class number

    Returns:
        models (dictionary): A dictionary contains rgb, flow and audio models.
    """
    model_rgb = model_flow = model_audio = None
    if rgb:
        model_rgb = ta3n(
            "rgb_ta3n", input_size, output_size, input_type, frame_aggregation, segments
        )
        model_all = model_rgb
    if flow:
        model_flow = ta3n(
            "flow_ta3n", input_size, output_size, progress, input_type=input_type, num_classes=dict_n_class
        )
        model_all = model_flow
    if audio:
        model_audio = ta3n(
            "audio_ta3n", input_size, output_size, progress, input_type=input_type, num_classes=dict_n_class
        )
        model_all = model_audio

    # For debugging
    if rgb and flow and audio:
        model_all = ta3n(
            "rgb_ta3n", pretrained, 3 * input_size, output_size, progress, input_type=input_type,
            num_classes=dict_n_class
        )
    return {"rgb": model_rgb, "flow": model_flow, "audio": model_audio, "all": model_all}


def get_classnet_ta3n(input_size_frame, input_size_video, dict_n_class, dropout_rate):
    frame_model = {
        "verb": ClassNetTA3NFrame(input_size=input_size_frame, output_size=dict_n_class["verb"]),
        "noun": ClassNetTA3NFrame(input_size=input_size_frame, output_size=dict_n_class["noun"]),
    }
    video_model = {
        "verb": ClassNetTA3NVideo(input_size=input_size_frame, output_size=dict_n_class["verb"],
                                  dropout_rate=dropout_rate),
        "noun": ClassNetTA3NVideo(input_size=input_size_frame, output_size=dict_n_class["noun"],
                                  dropout_rate=dropout_rate),
    }
    return {"frame-level": frame_model, "video-level": video_model}


def get_domainnet_ta3n(input_size_frame, input_size_video):
    frame_model = DomainNetTA3N(input_size=input_size_frame)
    video_model = DomainNetTA3N(input_size=input_size_frame)
    return {"frame-level": frame_model, "video-level": video_model}

#
# model_urls = {
#     "rgb_boring": None,
#     "flow_boring": None,
#     "audio_boring": None,
# }
#
#
# class BoringNetVideo(nn.Module):
#     """Regular simple network for video input.
#
#     Args:
#         input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
#         n_channel (int, optional): the number of channel for Linear and BN layers.
#         dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
#     """
#
#     def __init__(self, input_size=512, n_channel=512, n_out=256, dropout_keep_prob=0.5):
#         super(BoringNetVideo, self).__init__()
#         self.hidden_sizes = 512
#         self.num_layers = 4
#
#         self.transformer = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     emb_dim=input_size,
#                     num_heads=8,
#                     att_dropout=0.1,
#                     att_resid_dropout=0.1,
#                     final_dropout=0.1,
#                     max_seq_len=9,
#                     ff_dim=self.hidden_sizes,
#                     causal=False,
#                 )
#                 for _ in range(self.num_layers)
#             ]
#         )
#
#         self.fc1 = nn.Linear(input_size, n_channel)
#         self.relu1 = nn.ReLU()
#         self.dp1 = nn.Dropout(dropout_keep_prob)
#         self.fc2 = nn.Linear(n_channel, n_out)
#         self.selayer1 = SELayer4feat(channel=8, reduction=2)
#
#         # self.dim_reduction_layer = torch.nn.Identity()
#         #
#         # self.classification_vector = nn.Parameter(torch.randn(1, 1, input_size))
#         # self.pos_encoding = nn.Parameter(
#         #     torch.randn(1, 9, input_size)
#         # )
#
#     def forward(self, x):
#         # (B, F, INPUT_DIM) -> (B, F, D)
#
#         # x = self.dim_reduction_layer(x)
#         # B, F, D = x.size()
#
#         # classification_vector = self.classification_vector.repeat((B, 1, 1))
#         # (B, F, D) -> (B, 1+F, D)
#         # x = torch.cat([classification_vector, x], dim=1)
#         # seq_len = x.size(1)
#         # for layer in self.transformer:
#         #     x = x + self.pos_encoding[:, :seq_len, :]
#         #     x = layer(x)
#         x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
#         x = self.selayer1(x)
#         return x
#
#
# def boring_net(name, pretrained=False, input_size=1024, n_out=256, progress=True):
#     """Get BoringNetVideo module with pretrained model."""
#     model = BoringNetVideo(input_size=input_size, n_out=n_out)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def boring_net_joint(
#         rgb_name=None, flow_name=None, audio_name=None, pretrained=False, input_size=1024, n_out=256, progress=True
# ):
#     """Get BoringNetVideo model for different inputs.
#
#     Args:
#         rgb_name (string, optional): the name of pre-trained model for rgb input.
#         flow_name (string, optional): the name of pre-trained model for flow input.
#         audio_name (string, optional): the name of pre-trained model for audio input.
#         pretrained (bool, optional): choose if pretrained parameters are used. (Default: False),
#         input_size (int, optional): dimension of the final feature vector. Defaults to 1024.
#         n_out (int, optional): dimension of the output feature vector. Defaults to 256.
#         progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)
#
#     Returns:
#         models (dictionary): A dictionary contains rgb, flow and audio models.
#     """
#     model_rgb = model_flow = model_audio = None
#     if rgb_name is not None:
#         model_rgb = boring_net(rgb_name, pretrained, input_size, n_out, progress)
#     if flow_name is not None:
#         model_flow = boring_net(flow_name, pretrained, input_size, n_out, progress)
#     if audio_name is not None:
#         model_audio = boring_net(audio_name, pretrained, input_size, n_out, progress)
#     return {"rgb": model_rgb, "flow": model_flow, "audio": model_audio}

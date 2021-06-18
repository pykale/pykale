from torch import nn
from torchvision.models.utils import load_state_dict_from_url

from kale.embed.video_selayer import SELayer4feat
from kale.embed.video_transformer import TransformerBlock

model_urls_ta3n = {
    "rgb_ta3n": None,
    "flow_ta3n": None,
    "audio_ta3n": None,
}


class TA3N(nn.Module):
    def __init__(
        self, input_size=512, output_size=512, dropout_rate=0.5, add_fc=2, bn_layer="trn", trn_bottle_neck=512
    ):
        super(TA3N, self).__init__()
        self.bn_layer = bn_layer
        self.add_fc = add_fc
        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=dropout_rate)
        output_size = input_size
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.fc3 = nn.Linear(output_size, output_size)
        self.bn_shared = nn.BatchNorm1d(output_size)
        self.bn_trn = nn.BatchNorm1d(trn_bottle_neck)
        self.bn_1 = nn.BatchNorm1d(output_size)
        self.bn_2 = nn.BatchNorm1d(output_size)

    def forward(self, input):
        x = input.view(-1, input.size()[-1])
        x = self.fc1(input)

        if self.bn_layer == "shared":
            x = self.bn_shared(x)
        elif "trn" in self.bn_layer:
            try:
                x = self.bn_trn(x)
            except (RuntimeError):
                pass
        elif self.bn_layer == "temconv_1":
            x = self.bn_1_s(x)
        elif self.bn_layer == "temconv_2":
            x = self.bn_2(x)

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


def ta3n(name, pretrained=False, input_size=1024, n_out=256, progress=True):
    """Get TA3N module with pretrained model."""
    model = TA3N(input_size=input_size)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls_ta3n[name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def ta3n_joint(
    rgb_name=None, flow_name=None, audio_name=None, pretrained=False, input_size=1024, n_out=256, progress=True
):
    """Get TA3N model for different inputs.

    Args:
        rgb_name (string, optional): the name of pre-trained model for rgb input.
        flow_name (string, optional): the name of pre-trained model for flow input.
        audio_name (string, optional): the name of pre-trained model for audio input.
        pretrained (bool, optional): choose if pretrained parameters are used. (Default: False),
        input_size (int, optional): dimension of the final feature vector. Defaults to 1024.
        n_out (int, optional): dimension of the output feature vector. Defaults to 256.
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        models (dictionary): A dictionary contains rgb, flow and audio models.
    """
    model_rgb = model_flow = model_audio = None
    if rgb_name is not None:
        model_rgb = boring_net(rgb_name, pretrained, input_size, n_out, progress)
    if flow_name is not None:
        model_flow = boring_net(flow_name, pretrained, input_size, n_out, progress)
    if audio_name is not None:
        model_audio = boring_net(audio_name, pretrained, input_size, n_out, progress)
    return {"rgb": model_rgb, "flow": model_flow, "audio": model_audio}


model_urls = {
    "rgb_boring": None,
    "flow_boring": None,
    "audio_boring": None,
}


class BoringNetVideo(nn.Module):
    """Regular simple network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
    """

    def __init__(self, input_size=512, n_channel=512, n_out=256, dropout_keep_prob=0.5):
        super(BoringNetVideo, self).__init__()
        self.hidden_sizes = 512
        self.num_layers = 4

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=input_size,
                    num_heads=8,
                    att_dropout=0.1,
                    att_resid_dropout=0.1,
                    final_dropout=0.1,
                    max_seq_len=9,
                    ff_dim=self.hidden_sizes,
                    causal=False,
                )
                for _ in range(self.num_layers)
            ]
        )

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
        # (B, F, INPUT_DIM) -> (B, F, D)

        # x = self.dim_reduction_layer(x)
        # B, F, D = x.size()

        # classification_vector = self.classification_vector.repeat((B, 1, 1))
        # (B, F, D) -> (B, 1+F, D)
        # x = torch.cat([classification_vector, x], dim=1)
        # seq_len = x.size(1)
        # for layer in self.transformer:
        #     x = x + self.pos_encoding[:, :seq_len, :]
        #     x = layer(x)
        x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
        x = self.selayer1(x)
        return x


def boring_net(name, pretrained=False, input_size=1024, n_out=256, progress=True):
    """Get BoringNetVideo module with pretrained model."""
    model = BoringNetVideo(input_size=input_size, n_out=n_out)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def boring_net_joint(
    rgb_name=None, flow_name=None, audio_name=None, pretrained=False, input_size=1024, n_out=256, progress=True
):
    """Get BoringNetVideo model for different inputs.

    Args:
        rgb_name (string, optional): the name of pre-trained model for rgb input.
        flow_name (string, optional): the name of pre-trained model for flow input.
        audio_name (string, optional): the name of pre-trained model for audio input.
        pretrained (bool, optional): choose if pretrained parameters are used. (Default: False),
        input_size (int, optional): dimension of the final feature vector. Defaults to 1024.
        n_out (int, optional): dimension of the output feature vector. Defaults to 256.
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        models (dictionary): A dictionary contains rgb, flow and audio models.
    """
    model_rgb = model_flow = model_audio = None
    if rgb_name is not None:
        model_rgb = boring_net(rgb_name, pretrained, input_size, n_out, progress)
    if flow_name is not None:
        model_flow = boring_net(flow_name, pretrained, input_size, n_out, progress)
    if audio_name is not None:
        model_audio = boring_net(audio_name, pretrained, input_size, n_out, progress)
    return {"rgb": model_rgb, "flow": model_flow, "audio": model_audio}

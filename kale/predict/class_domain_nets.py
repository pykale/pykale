# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk or xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Classification of data or domain

Modules for typical classification tasks (into class labels) and
adversarial discrimination of source vs target domains, from
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/modules.py
"""

import torch
import torch.nn as nn

from kale.embed.video_i3d import Unit3D
from kale.loaddata.video_access import get_class_type
from kale.pipeline.video_domain_adapter import ReverseLayerF


# Previously FFSoftmaxClassifier
class SoftmaxNet(nn.Module):
    """Regular and domain classifier network for regular-size images

    Args:
        input_dim (int, optional): the dimension of the final feature vector.. Defaults to 15.
        n_classes (int, optional): the number of classes. Defaults to 2.
        name (str, optional): the classifier name. Defaults to "c".
        hidden (tuple, optional): the hidden layer sizes. Defaults to ().
        activation_fn ([type], optional): the activation function. Defaults to nn.ReLU.
    """

    def __init__(
        self, input_dim=15, n_classes=2, name="c", hidden=(), activation_fn=nn.ReLU, **activation_args,
    ):

        super(SoftmaxNet, self).__init__()
        self._n_classes = n_classes
        self._activation_fn = activation_fn
        self.chain = nn.Sequential()
        self.name = name
        self._hidden_sizes = hidden if hidden is not None else ()
        last_dim = input_dim
        for i, h in enumerate(self._hidden_sizes):
            self.chain.add_module(f"{name}_fc{i}", nn.Linear(last_dim, h))
            self.chain.add_module(f"f_{activation_fn.__name__}{i}", activation_fn(**activation_args))
            last_dim = h
        self.chain.add_module(f"{name}_fc_last", nn.Linear(last_dim, self._n_classes))
        self.activation = nn.LogSoftmax(dim=1)
        self.loss_class = nn.NLLLoss()

    def forward(self, input_data):
        class_output = self.chain(input_data)
        return class_output

    def extra_repr(self):
        if len(self._hidden_sizes) > 0:
            return f"{self.name}: {self.hidden_sizes}x{self._activation_fn.__name__}xLin"
        return f"{self.name}: Linear"

    def n_classes(self):
        return self._n_classes


# Previously DataClassifierDigits
class ClassNetSmallImage(nn.Module):
    """Regular classifier network for small-size images

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 128.
        n_class (int, optional): the number of classes. Defaults to 10.
    """

    def __init__(self, input_size=128, n_class=10):
        super(ClassNetSmallImage, self).__init__()
        self._n_classes = n_class
        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout2d()
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, n_class)

    def n_classes(self):
        return self._n_classes

    def forward(self, input):
        x = self.dp1(self.relu1(self.bn1(self.fc1(input))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Previously DomainClassifierDigits
class DomainNetSmallImage(nn.Module):
    """Domain classifier network for small-size images

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 128.
        bigger_discrim (bool, optional): whether to use deeper network. Defaults to False.
    """

    def __init__(self, input_size=128, bigger_discrim=False):

        super(DomainNetSmallImage, self).__init__()
        output_size = 500 if bigger_discrim else 100

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, 100) if bigger_discrim else nn.Linear(output_size, 2)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, input):
        x = self.relu1(self.bn1(self.fc1(input)))
        if self.bigger_discrim:
            x = self.relu2(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x


# For Video/Action Recognition, DataClassifier.
class ClassNetVideo(nn.Module):
    """Regular classifier network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
        dict_n_class (dict, optional): the dictionary of class number for specific dataset.
    """

    def __init__(
        self,
        input_size=512,
        n_verb_channel=256,
        n_noun_channel=512,
        dropout_keep_prob=0.5,
        dict_n_class=8,
        class_type="verb",
    ):
        super(ClassNetVideo, self).__init__()
        self.verb, self.noun = get_class_type(class_type)
        if self.verb:
            self.n_verb_class = dict_n_class["verb"]
            self.fc1 = nn.Linear(input_size, n_verb_channel)
            self.bn1 = nn.BatchNorm1d(n_verb_channel)
            self.relu1 = nn.ReLU()
            self.dp1 = nn.Dropout(dropout_keep_prob)
            self.fc11 = nn.Linear(n_verb_channel, self.n_verb_class)
        if self.noun:
            self.n_noun_class = dict_n_class["noun"]
            self.fc2 = nn.Linear(input_size, n_noun_channel)
            self.bn2 = nn.BatchNorm1d(n_noun_channel)
            self.relu2 = nn.ReLU()
            self.dp2 = nn.Dropout(dropout_keep_prob)
            self.fc21 = nn.Linear(n_noun_channel, self.n_noun_class)

    def forward(self, input):
        x_verb = self.fc11(self.dp1(self.relu1(self.bn1(self.fc1(input)))))
        if self.verb and not self.noun:
            x_noun = None
        if self.verb and self.noun:
            x_noun = self.fc21(self.dp2(self.relu2(self.bn2(self.fc2(input)))))
        return [x_verb, x_noun]


class ClassNetVideoConv(nn.Module):
    """Classifier network for video input refer to MMSADA.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 1024.
        n_class (int, optional): the number of classes. Defaults to 8.

    References:
        Munro Jonathan, and Dima Damen. "Multi-modal domain adaptation for fine-grained action recognition."
        In CVPR, pp. 122-132. 2020.
    """

    def __init__(self, input_size=1024, n_class=8):
        super(ClassNetVideoConv, self).__init__()
        self.dp = nn.Dropout()
        self.logits = Unit3D(
            in_channels=input_size,
            output_channels=n_class,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
        )

    def forward(self, input):
        x = self.logits(self.dp(input))
        return x


# For Video/Action Recognition, DomainClassifier.
class DomainNetVideo(nn.Module):
    """Regular domain classifier network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
    """

    def __init__(self, input_size=128, n_channel=100, class_type="verb"):
        super(DomainNetVideo, self).__init__()

        self.fc1 = nn.Linear(input_size, n_channel)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(n_channel, 2)

    def forward(self, input):
        x = self.relu1(self.bn1(self.fc1(input)))
        x = self.fc2(x)
        return x


class ClassNetTA3N(nn.Module):
    """Regular Classifier network for TA3N.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
    """

    def __init__(self, input_size=128, n_channel=100, class_type="verb", train_segments=5):
        super(DomainNetVideo, self).__init__()
        self.fc_feature_domain = nn.Linear(input_size, n_channel)
        self.fc_classifier_domain = nn.Linear(n_channel, 2)
        self.relu = nn.ReLU(inplace=True)
        self.relation_domain_classifier_all = nn.ModuleList()
        self.train_segments = train_segments
        for i in range(self.train_segments - 1):
            relation_domain_classifier = nn.Sequential(
                nn.Linear(input_size, n_channel), nn.ReLU(), nn.Linear(n_channel, 2)
            )
            self.relation_domain_classifier_all += [relation_domain_classifier]

    def forward(self, input, beta=0, isRelation=False):
        prediction_video = 0
        return prediction_video


class DomainNetTA3N(nn.Module):
    """Regular domain classifier network for TA3N.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
    """

    def __init__(self, input_size=128, n_channel=100, class_type="verb", train_segments=5):
        super(DomainNetVideo, self).__init__()
        self.fc_feature_domain = nn.Linear(input_size, n_channel)
        self.fc_classifier_domain = nn.Linear(n_channel, 2)
        self.relu = nn.ReLU(inplace=True)
        self.relation_domain_classifier_all = nn.ModuleList()
        self.train_segments = train_segments
        for i in range(self.train_segments - 1):
            relation_domain_classifier = nn.Sequential(
                nn.Linear(input_size, n_channel), nn.ReLU(), nn.Linear(n_channel, 2)
            )
            self.relation_domain_classifier_all += [relation_domain_classifier]

    def forward(self, input, beta=0, isRelation=False):
        if not isRelation:
            x = ReverseLayerF.apply(input, beta)
            x = self.fc_feature_domain(x)
            x = self.relu(x)
            prediction = self.fc_classifier_domain(x)
            return prediction
        else:
            # 128x4x256 --> (128x4)x2
            prediction_video = None
            for i in range(len(self.relation_domain_classifier_all)):
                x_relation = input[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
                x_relation = ReverseLayerF.apply(x_relation, beta)

                prediction_single = self.relation_domain_classifier_all[i](x_relation)

                if prediction_video is None:
                    prediction_video = prediction_single.view(-1, 1, 2)
                else:
                    prediction_video = torch.cat((prediction_video, prediction_single.view(-1, 1, 2)), 1)

            prediction_video = prediction_video.view(-1, 2)

            return prediction_video

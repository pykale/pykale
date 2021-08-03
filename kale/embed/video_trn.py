# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Python implementation of Temporal Relation Network (TRN) for TA3N model.
"""

import torch
import numpy as np

from torch import nn
from kale.pipeline.domain_adapter import GradReverse


class TRNRelationModule(nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(TRNRelationModule, self).__init__()
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        # naive concatenate
        classifier = nn.Sequential(
            nn.ReLU(), nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck), nn.ReLU(),
        )
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames * self.img_feature_dim)
        input = self.classifier(input)
        return input


class TRNRelationModuleMultiScale(nn.Module):
    # Temporal Relation module in multiply scale, summing over [2-frame relation, 3-frame relation, ...,
    # n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(TRNRelationModuleMultiScale, self).__init__()
        self.subsample_num = 3  # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]  # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(
                min(self.subsample_num, len(relations_scale))
            )  # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList()  # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(nn.ReLU(), nn.Linear(scale * self.img_feature_dim, num_bottleneck), nn.ReLU(),)

            self.fc_fusion_scales += [fc_fusion]

        # print("Multi-Scale Temporal Relation Network Module in use", ["%d-frame relation" % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0], :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1)  # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [
                int(np.ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)
            ]

            # for idx in idx_relations_randomsample:
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools

        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


class TemporalAttention(nn.Module):
    """This is the novel temporal attention module of TA3N.
    It TRN to effectively capture temporal information.
    Uses extra domain classifiers to add temporal attention.
    """

    def __init__(
        self, input_size=512, num_segments=5, beta=0.75, trn_bottleneck=256, use_attn="TransAttn",
    ):
        super(TemporalAttention, self).__init__()
        self.num_segments = num_segments
        self.beta = beta
        self.use_attn = use_attn

        self.relu = nn.ReLU(inplace=True)

        self.TRN = TRNRelationModuleMultiScale(input_size, trn_bottleneck, self.num_segments)

        self.relation_domain_classifier_all = nn.ModuleList()
        for i in range(num_segments - 1):
            relation_domain_classifier = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
            self.relation_domain_classifier_all += [relation_domain_classifier]
        self.dropout_v = nn.Dropout(p=0.5)

        self.fc_1_domain_frame = nn.Linear(input_size, input_size)
        self.fc_2_domain_frame = nn.Linear(input_size, 2)

    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy

        return weights

    def get_general_attn(self, feat):
        num_segments = feat.size()[1]
        feat = feat.view(-1, feat.size()[-1])  # reshape features: 128x4x256 --> (128x4)x256
        weights = self.attn_layer(feat)  # e.g. (128x4)x1
        weights = weights.view(-1, num_segments, weights.size()[-1])  # reshape attention weights: (128x4)x1 --> 128x4x1
        weights = nn.functional.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

        return weights

    def domain_classify_relation(self, input, beta):
        prediction_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            x = input[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
            x = GradReverse.apply(x, beta)
            prediction_single = self.relation_domain_classifier_all[i](x)

            if prediction_video is None:
                prediction_video = prediction_single.view(-1, 1, 2)
            else:
                prediction_video = torch.cat((prediction_video, prediction_single.view(-1, 1, 2)), 1)

        prediction_video = prediction_video.view(-1, 2)
        prediction_video = prediction_video.detach().clone()

        return prediction_video

    def domain_classifier_frame(self, input, beta):
        x = GradReverse.apply(input, beta)
        x = self.fc_1_domain_frame(x)
        x = self.relu(x)
        x = self.fc_2_domain_frame(x)

        return x

    def get_attention(self, input, pred):
        """Adds attention / excites the input features based on temporal pooling
        """
        if self.use_attn == "TransAttn":
            weights = self.get_trans_attn(pred)
        elif self.use_attn == "general":
            weights = self.get_general_attn(input)
        while len(list(weights.size())) < len(list(input.size())):
            weights = weights.unsqueeze(-1)
        if weights.size()[0] < input.size()[0]:
            weights = weights.view(input.size()[0], 2, -1)
        repeats = [input.size()[i] // weights.size()[i] + 1 for i in range(0, len(list(input.size())))]
        weights = weights.repeat(repeats)  # reshape & repeat weights (e.g. 16 x 4 x 256)
        weights = weights[: input.size()[0], : input.size()[1], : input.size()[2]]
        return weights

    def forward(self, input, beta):
        domain_pred_0 = self.domain_classifier_frame(input, beta[2])
        domain_pred_0 = domain_pred_0.view((-1,) + domain_pred_0.size()[-1:])
        # reshape based on the segments (e.g. 640x512 --> 128x5x512)
        x = input.view((-1, self.num_segments) + input.size()[-1:])
        x = self.TRN(x)
        # adversarial branch
        domain_pred_relation = self.domain_classify_relation(x, beta[0])
        domain_pred_1 = domain_pred_relation.view((-1, x.size()[1]) + domain_pred_relation.size()[-1:])
        # adding attention
        weights_attn = self.get_attention(x, domain_pred_relation)
        x = (weights_attn + 1) * x

        x = torch.sum(x, 1)
        x = self.dropout_v(x)
        return x, domain_pred_0, domain_pred_1
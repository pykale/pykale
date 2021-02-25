import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from kale.embed.video_res3d import r3d_18
from kale.embed.video_i3d import InceptionI3d
from kale.predict.class_domain_nets import DomainNetVideo, ClassNetVideo
from kale.pipeline.domain_adapter import ReverseLayerF

cuda0 = torch.device('cuda:0')

os.environ["PYTHONHASHSEED"] = str(2020)
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
# torch.cuda.manual_seed(2020)
# torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)


class Res(nn.Module):
    def __init__(self):
        super(Res, self).__init__()
        self.ext = r3d_18()
        self.classifier = ClassNetVideo(input_size=512)
        self.domain = DomainNetVideo(input_size=512)

    def forward(self, x):
        feat = self.ext(x)
        feat1 = feat.view(feat.size(0), -1)
        class_output = self.classifier(feat1)

        reverse_feature = ReverseLayerF.apply(feat1, 1.0)
        adversarial_output = self.domain(reverse_feature)
        return x, class_output, adversarial_output


x1 = torch.rand([2, 3, 2, 224, 224], device=cuda0)

yc = torch.ones([2, 8], device=cuda0)
yd = torch.ones([2, 2], device=cuda0)

model = Res().cuda()

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01, weight_decay=0.0005, nesterov=True)

l1 = []
l2 = []
l3 = []
out = []

n = 1

for epoch in range(100):
    print(epoch)
    x, oc, oadv = model(x1)

    opt.zero_grad()

    loss1 = F.binary_cross_entropy_with_logits(oc, yc)
    loss2 = F.binary_cross_entropy_with_logits(oadv, yd)

    l1.append(loss1.item())
    l2.append(loss2.item())
    loss = loss1 + loss2
    l3.append(loss.item())
    # out.append(o12.tolist())

    loss.backward()

    opt.step()

print(l1)
print(l2)
print(l3)
# print(out)

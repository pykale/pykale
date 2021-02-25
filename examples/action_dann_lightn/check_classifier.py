import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from kale.predict.class_domain_nets import DomainNetVideo, ClassNetVideo
from kale.embed.video_i3d import InceptionI3d

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


x1 = torch.rand([16, 1024], device=cuda0)

y1 = torch.ones(16, 2, device=cuda0)

model1 = DomainNetVideo(input_size=1024).cuda()
# model1 = ClassNetVideo(input_size=1024).cuda()

opt1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.01, weight_decay=0.0005, nesterov=True)

l = []
out = []

n = 1

for epoch in range(100):
    print(epoch)
    o1 = model1(x1)

    opt1.zero_grad()

    loss1 = F.binary_cross_entropy_with_logits(o1, y1)

    l.append(loss1.item())
    out.append(o1.tolist())

    loss1.backward()

    opt1.step()

print(l)
print(out)

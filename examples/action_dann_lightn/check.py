import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from kale.embed.video_res3d import r3d_18

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


x1 = torch.rand([2, 3, 2, 224, 224], device=cuda0)

y1 = torch.ones([2, 8], device=cuda0)

model1 = r3d_18().cuda()

opt1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.01, weight_decay=0.0005, nesterov=True)

l = []
out = []

n = 1

for epoch in range(100):
    print(epoch)
    o1 = model1(x1).squeeze()
    o11 = o1.view(o1.size(0), -1)
    o12 = nn.Linear(512, 8, bias=False).cuda()(o11)

    opt1.zero_grad()

    loss1 = F.binary_cross_entropy_with_logits(o12, y1)

    l.append(loss1.item())
    out.append(o12.tolist())

    loss1.backward()

    opt1.step()

print(l)
print(out)

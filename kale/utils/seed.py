# Created by Haiping Lu based on https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/6
# Ref: https://pytorch.org/docs/stable/notes/randomness.html
import os
import random
import torch
import numpy as np

# Results can be software/hardware-dependent
# Exactly reproduciable results are expected only on the same software and hardware
def set_seed(seed=1000):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

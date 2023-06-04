import pytest
import torch
import torch.nn as nn

from kale.utils.initialize_nn import bias_init, xavier_init
from kale.utils.seed import set_seed


def test_xavier_init():
    set_seed(2022)
    module = nn.Linear(10, 20)
    xavier_init(module)
    assert torch.var(module.weight).item() == pytest.approx(2.0 / (module.in_features + module.out_features), rel=1e-1)


def test_bias_init():
    module = nn.Linear(10, 20)
    bias_init(module)
    assert torch.mean(module.bias).item() == pytest.approx(0.0, rel=1e-2)
    assert torch.var(module.bias).item() == pytest.approx(0.0, rel=1e-2)

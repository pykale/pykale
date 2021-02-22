import random

import numpy as np
import pytest
import torch
from numpy import testing

from kale.utils.seed import set_seed


@pytest.fixture
def base_rand():
    return 0.7773566427005639


def test_set_seed_base(base_rand):
    set_seed()
    result = random.random()
    testing.assert_equal(result, base_rand)


@pytest.fixture
def np_rand():
    return 0.6535895854646095


def test_set_seed_numpy(np_rand):
    set_seed()
    result = np.random.rand()
    testing.assert_equal(result, np_rand)


@pytest.fixture
def torch_rand():
    return 0.3189346194267273


def test_set_seed_torch(torch_rand):
    set_seed()
    result = torch.rand(1).item()
    testing.assert_equal(result, torch_rand)

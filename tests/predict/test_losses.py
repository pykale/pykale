import numpy as np
import pytest

from kale.predict.rdc import rdc


@pytest.mark.parametrize("n", [100, 200])
def test_rdc(n):
    x = np.random.random((n, 30))
    y = np.random.random((n, 40))
    ind_coef = rdc(x, y)
    assert ind_coef >= 0
    assert ind_coef <= 1

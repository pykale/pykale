import logging

import pandas as pd
import pytest

from kale.prepdata.tabular_transform import apply_confidence_inversion
from kale.utils.seed import set_seed

seed = 36
set_seed(seed)

LOGGER = logging.getLogger(__name__)


DUMMY_DICT = pd.DataFrame({"data": [0.1, 0.2, 0.9, 1.5]})


@pytest.mark.parametrize("input, expected", [(DUMMY_DICT, [1 / 0.1, 1 / 0.2, 1 / 0.9, 1 / 1.5])])
def test_apply_confidence_inversion(input, expected):

    # test that it inverts correctly
    assert list(apply_confidence_inversion(input, "data")["data"]) == pytest.approx(expected)

    # test that a KeyError is raised successfully if key not in dict.
    with pytest.raises(KeyError, match=r".* key .*"):
        apply_confidence_inversion({}, "data") == pytest.approx(expected)

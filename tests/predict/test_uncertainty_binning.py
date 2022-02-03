import logging

import pytest

from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.utils.seed import set_seed

seed = 36
set_seed(seed)

LOGGER = logging.getLogger(__name__)

DUMMY_TABULAR_DATA = {
    "uid": ["PHD_2154", "PHD_2158", "PHD_217", "PHD_2194"],
    "E-CPV Error": [1.4142135, 3.1622777, 5.0990195, 61.846584],
    "E-CPV Uncertainty": [4.25442667, 4.449976897, 1.912124681, 35.76085777],
    "E-MHA Error": [3.1622777, 3.1622777, 4, 77.00649],
    "E-MHA Uncertainty": [0.331125357, 0.351173535, 1.4142135, 0.142362904],
    "S-MHA Error": [3.1622777, 1.4142135, 5.0990195, 56.32051],
    "S-MHA Uncertainty": [0.500086973, 0.235296882, 1.466040241, 0.123874651],
    "Validation Fold": [1, 1, 1, 1],
    "Testing Fold": [0, 0, 0, 0],
}


class TestQuantileBinningPredictions:
    @pytest.mark.parametrize(
        "uncertainty_thresh_list, expected",
        [
            ([[1], [3], [5], [9]], {"PHD_2154": 2, "PHD_2158": 2, "PHD_217": 1, "PHD_2194": 4}),
            ([[1], [1.5], [1.6], [2]], {"PHD_2154": 4, "PHD_2158": 4, "PHD_217": 3, "PHD_2194": 4}),
            ([[40], [42], [43], [44]], {"PHD_2154": 0, "PHD_2158": 0, "PHD_217": 0, "PHD_2194": 0}),
        ],
    )
    def test_quantile_binning_predictions_thresh(self, uncertainty_thresh_list, expected):

        test_dict = dict(zip(DUMMY_TABULAR_DATA["uid"], DUMMY_TABULAR_DATA["E-CPV Uncertainty"]))
        assert quantile_binning_predictions(test_dict, uncertainty_thresh_list) == expected

    # test wrong dict format
    def test_dict_format(self):
        with pytest.raises(ValueError, match="uncertainties_test must be of type dict"):
            quantile_binning_predictions([1, 2, 3], [[1], [1.5], [1.6], [2]])
        with pytest.raises(ValueError, match=r"Dict uncertainties_test should be of structure .*"):
            quantile_binning_predictions(
                {"PHD_2154": "2", "PHD_2158": 2, "PHD_217": 1, "PHD_2194": 4}, [[1], [1.5], [1.6], [2]]
            )
        with pytest.raises(ValueError, match=r"uncert_thresh list should be 2D .*"):
            quantile_binning_predictions({"PHD_2154": 2, "PHD_2158": 2, "PHD_217": 1, "PHD_2194": 4}, [1, 1.5, 1.6, 2])

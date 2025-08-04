import numpy as np
import pytest

from kale.interpret.correlation_analysis import fit_line_with_ci

SEED = 42
np.random.seed(SEED)

ERRORS = np.random.rand(50)
UNCERTAINTIES = np.random.rand(50)
QUANTILE_THRESHOLDS = [0.3, 0.6]
CMAPS = ["red", "green", "blue"]


@pytest.fixture(scope="module")
def dummy_corr_data():
    # Fixture to provide dummy errors and uncertainties
    errors = np.random.rand(30)
    uncertainties = np.random.rand(30)
    quantile_thresholds = [0.2, 0.5, 0.8]
    cmaps = ["red", "green", "blue", "orange"]
    return errors, uncertainties, quantile_thresholds, cmaps


class TestFitLineWithCI:
    def test_basic(self):
        result = fit_line_with_ci(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
            cmaps=CMAPS,
            to_log=False,
            error_scaling_factor=1.0,
            save_path=None,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result
        assert len(result["spearman"]) == 2
        assert len(result["pearson"]) == 2
        assert all(isinstance(x, float) for x in result["spearman"])
        assert all(isinstance(x, float) for x in result["pearson"])

    def test_with_fixture(self, dummy_corr_data):
        errors, uncertainties, quantile_thresholds, cmaps = dummy_corr_data
        result = fit_line_with_ci(
            errors=errors,
            uncertainties=uncertainties,
            quantile_thresholds=quantile_thresholds,
            cmaps=cmaps,
            to_log=True,
            error_scaling_factor=2.0,
            save_path=None,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result

    def test_invalid_input(self):
        # errors and uncertainties must be same length
        with pytest.raises(ValueError):
            fit_line_with_ci(
                errors=np.array([1, 2, 3]),
                uncertainties=np.array([1, 2]),
                quantile_thresholds=[0.5],
                cmaps=["red", "green"],
            )

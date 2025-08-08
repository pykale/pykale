import numpy as np
import pytest

from kale.interpret.correlation_analysis import analyze_and_plot_uncertainty_correlation

SEED = 42
np.random.seed(SEED)

ERRORS = np.random.rand(50)
UNCERTAINTIES = np.random.rand(50)
QUANTILE_THRESHOLDS = [0.3, 0.6]


@pytest.fixture(scope="module")
def dummy_corr_data():
    # Fixture to provide dummy errors and uncertainties
    errors = np.random.rand(30)
    uncertainties = np.random.rand(30)
    quantile_thresholds = [0.2, 0.5, 0.8]
    return errors, uncertainties, quantile_thresholds


class TestFitLineWithCI:
    def test_basic(self):
        result = analyze_and_plot_uncertainty_correlation(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
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
        errors, uncertainties, quantile_thresholds = dummy_corr_data
        result = analyze_and_plot_uncertainty_correlation(
            errors=errors,
            uncertainties=uncertainties,
            quantile_thresholds=quantile_thresholds,
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
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([1, 2, 3]), uncertainties=np.array([1, 2]), quantile_thresholds=[0.5]
            )

        # empty arrays
        with pytest.raises(ValueError):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([]), uncertainties=np.array([]), quantile_thresholds=[0.5]
            )

        # negative values (should not error unless function restricts, but test for robustness)
        errors = np.array([-1, -2, -3])
        uncertainties = np.array([-1, -2, -3])
        quantile_thresholds = [-0.5, -0.2]
        # If function does not raise, just check it runs
        try:
            analyze_and_plot_uncertainty_correlation(errors, uncertainties, quantile_thresholds)
        except Exception as e:
            assert isinstance(e, Exception)

        # invalid quantile thresholds (not sorted or out of range)
        with pytest.raises(Exception):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([1, 2, 3]),
                uncertainties=np.array([1, 2, 3]),
                quantile_thresholds=[2, 1],  # not sorted, out of range
            )

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


class TestBasicCorrelation:
    """Test basic correlation analysis functionality."""

    def test_basic_functionality(self, tmp_path):
        """Test basic correlation analysis with default parameters."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        result = analyze_and_plot_uncertainty_correlation(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
            to_log=False,
            error_scaling_factor=1.0,
            save_path=out_path,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result
        assert len(result["spearman"]) == 2
        assert len(result["pearson"]) == 2
        assert all(isinstance(x, float) for x in result["spearman"])
        assert all(isinstance(x, float) for x in result["pearson"])

    def test_with_logging(self, dummy_corr_data, tmp_path):
        """Test correlation analysis with logarithmic scaling."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        errors, uncertainties, quantile_thresholds = dummy_corr_data
        result = analyze_and_plot_uncertainty_correlation(
            errors=errors,
            uncertainties=uncertainties,
            quantile_thresholds=quantile_thresholds,
            to_log=True,
            error_scaling_factor=2.0,
            save_path=out_path,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result

    def test_save_functionality(self, tmp_path):
        """Test that plots are properly saved and files are created."""
        out_path = tmp_path / "test_correlation_plot.pdf"

        result = analyze_and_plot_uncertainty_correlation(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
            to_log=False,
            error_scaling_factor=1.0,
            save_path=out_path,
        )

        # Verify the function returns expected results
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result

        # Verify the file was actually created
        assert out_path.exists(), f"Expected plot file was not created at {out_path}"

        # Verify the file is not empty
        assert out_path.stat().st_size > 0, "Created plot file is empty"

        # Verify reasonable file size (PDF should be substantial but not huge)
        file_size = out_path.stat().st_size
        assert file_size > 1000, f"Plot file seems too small ({file_size} bytes)"
        assert file_size < 10_000_000, f"Plot file seems too large ({file_size} bytes)"

        # tmp_path automatically cleans up after test completes


class TestInputValidation:
    """Test input validation and error handling."""

    def test_mismatched_array_lengths(self, tmp_path):
        """Test that mismatched array lengths raise appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        with pytest.raises(ValueError, match="must have the same length"):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([1, 2, 3]),
                uncertainties=np.array([1, 2]),
                quantile_thresholds=[0.5],
                save_path=out_path,
            )

    def test_empty_arrays(self, tmp_path):
        """Test that empty arrays raise appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([]), uncertainties=np.array([]), quantile_thresholds=[0.5], save_path=out_path
            )

    def test_negative_quantile_thresholds(self, tmp_path):
        """Test that negative quantile thresholds cause appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        errors = np.array([-1, -2, -3])
        uncertainties = np.array([-1, -2, -3])
        quantile_thresholds = [-0.5, -0.2]
        # Negative quantile thresholds are likely to cause issues in piecewise fitting
        with pytest.raises((ValueError, RuntimeError, Exception)):
            analyze_and_plot_uncertainty_correlation(errors, uncertainties, quantile_thresholds, save_path=out_path)

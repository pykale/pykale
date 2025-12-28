import logging
from unittest.mock import patch

import numpy as np
import pytest

from kale.interpret.box_plot import BoxPlotData, create_boxplot_config, create_boxplot_data
from kale.interpret.uncertainty_quantiles import (
    plot_comparing_q_boxplot,
    plot_generic_boxplot,
    plot_per_model_boxplot,
)
from kale.loaddata.tabular_access import load_csv_columns

# from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

# import os
LOGGER = logging.getLogger(__name__)


seed = 36
set_seed(seed)

ERRORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UNCERTAINTIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


@pytest.fixture(scope="module")
def dummy_test_data(landmark_uncertainty_dl):
    dummy_tabular_data_dict = load_csv_columns(
        landmark_uncertainty_dl[1], "Testing Fold", np.arange(0), cols_to_return="All"
    )

    dummy_errors = dummy_tabular_data_dict["S-MHA Error"].values
    dummy_uncertainties = dummy_tabular_data_dict["S-MHA Error"].values

    return dummy_errors, dummy_uncertainties


# Additional test fixtures for plot testing
@pytest.fixture
def sample_boxplot_data():
    """Create sample data for boxplot testing."""
    evaluation_data = {
        "ResNet50_epistemic": [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]],
        "VGG16_epistemic": [[0.12, 0.22, 0.32], [0.17, 0.27, 0.37], [0.22, 0.32, 0.42]],
        "ResNet50_aleatoric": [[0.05, 0.15, 0.25], [0.1, 0.2, 0.3], [0.15, 0.25, 0.35]],
        "VGG16_aleatoric": [[0.07, 0.17, 0.27], [0.12, 0.22, 0.32], [0.17, 0.27, 0.37]],
    }
    return create_boxplot_data(
        evaluation_data_by_bins=[evaluation_data],
        uncertainty_categories=[["epistemic"], ["aleatoric"]],
        models=["ResNet50", "VGG16"],
        category_labels=["B1", "B2", "B3"],
        num_bins=3,
    )


@pytest.fixture
def sample_boxplot_config():
    """Create sample config for boxplot testing."""
    return create_boxplot_config(
        x_label="Test Bins",
        y_label="Test Error (%)",
        show=False,
        save_path=None,
    )


@pytest.fixture
def sample_comparing_q_data():
    """Create sample data for Q-value comparison testing."""
    q_data = [
        {"model_epistemic": [[0.1, 0.2], [0.15, 0.25]]},  # Q=5
        {"model_epistemic": [[0.09, 0.19], [0.14, 0.24]]},  # Q=10
        {"model_epistemic": [[0.08, 0.18], [0.13, 0.23]]},  # Q=15
    ]
    return create_boxplot_data(
        evaluation_data_by_bins=q_data,
        uncertainty_categories=[["epistemic"]],
        models=["model"],
        category_labels=["Q=5", "Q=10", "Q=15"],
        num_bins=2,
    )


class TestPlotFunctions:
    """Test the plotting wrapper functions."""

    @patch("kale.interpret.box_plot.GenericBoxPlotter.draw_boxplot")
    def test_plot_generic_boxplot(self, mock_draw, sample_boxplot_data, sample_boxplot_config):
        """Test plot_generic_boxplot function."""
        plot_generic_boxplot(sample_boxplot_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    @patch("kale.interpret.box_plot.PerModelBoxPlotter.draw_boxplot")
    def test_plot_per_model_boxplot(self, mock_draw, sample_boxplot_data, sample_boxplot_config):
        """Test plot_per_model_boxplot function."""
        plot_per_model_boxplot(sample_boxplot_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    @patch("kale.interpret.box_plot.ComparingQBoxPlotter.draw_boxplot")
    def test_plot_comparing_q_boxplot(self, mock_draw, sample_comparing_q_data, sample_boxplot_config):
        """Test plot_comparing_q_boxplot function."""
        plot_comparing_q_boxplot(sample_comparing_q_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    def test_plot_comparing_q_boxplot_missing_data(self, sample_boxplot_config):
        """Test that plot_comparing_q_boxplot raises error with missing data."""
        incomplete_data = BoxPlotData(evaluation_data_by_bins=[{}])
        with pytest.raises(ValueError, match="For comparing_q plots"):
            plot_comparing_q_boxplot(incomplete_data, sample_boxplot_config)

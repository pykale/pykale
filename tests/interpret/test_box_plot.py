from unittest.mock import MagicMock, patch

import pytest

# Import only the core components to avoid dependency issues
try:
    from kale.interpret.box_plot import (
        _calculate_spacing_adjustment,
        _create_data_item,
        _extract_bin_data,
        _find_data_key,
        BoxPlotConfig,
        BoxPlotData,
        ComparingQBoxPlotter,
        create_boxplot_config,
        create_boxplot_data,
        GenericBoxPlotter,
        PerModelBoxPlotter,
        SampleInfoMode,
    )
except ImportError:
    pytest.skip("Skipping box_plot tests due to missing dependencies", allow_module_level=True)


# Test fixtures for sample data
@pytest.fixture
def sample_evaluation_data():
    """Create sample evaluation data for testing."""
    return {
        "ResNet50_epistemic": [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]],
        "VGG16_epistemic": [[0.12, 0.22, 0.32], [0.17, 0.27, 0.37], [0.22, 0.32, 0.42]],
        "ResNet50_aleatoric": [[0.05, 0.15, 0.25], [0.1, 0.2, 0.3], [0.15, 0.25, 0.35]],
        "VGG16_aleatoric": [[0.07, 0.17, 0.27], [0.12, 0.22, 0.32], [0.17, 0.27, 0.37]],
    }


@pytest.fixture
def sample_q_comparison_data():
    """Create sample Q-value comparison data for testing."""
    return [
        {"model_epistemic": [[0.1, 0.2], [0.15, 0.25]]},  # Q=5
        {"model_epistemic": [[0.09, 0.19], [0.14, 0.24]]},  # Q=10
        {"model_epistemic": [[0.08, 0.18], [0.13, 0.23]]},  # Q=15
    ]


@pytest.fixture
def sample_config():
    """Create sample BoxPlotConfig for testing."""
    return create_boxplot_config(
        x_label="Test X Label",
        y_label="Test Y Label",
        colormap="Set1",
        show=False,  # Don't show plots during testing
        save_path=None,  # Don't save plots during testing
    )


@pytest.fixture
def sample_generic_data(sample_evaluation_data):
    """Create sample BoxPlotData for generic plotting."""
    return create_boxplot_data(
        evaluation_data_by_bins=[sample_evaluation_data],
        uncertainty_categories=[["epistemic"], ["aleatoric"]],
        models=["ResNet50", "VGG16"],
        category_labels=["B1", "B2", "B3"],
        num_bins=3,
    )


@pytest.fixture
def sample_per_model_data(sample_evaluation_data):
    """Create sample BoxPlotData for per-model plotting."""
    return create_boxplot_data(
        evaluation_data_by_bins=[sample_evaluation_data],
        uncertainty_categories=[["epistemic"], ["aleatoric"]],
        models=["ResNet50"],
        category_labels=["B1", "B2", "B3"],
        num_bins=3,
    )


@pytest.fixture
def sample_comparing_q_data(sample_q_comparison_data):
    """Create sample BoxPlotData for Q-value comparison."""
    return create_boxplot_data(
        evaluation_data_by_bins=sample_q_comparison_data,
        uncertainty_categories=[["epistemic"]],
        models=["model"],
        category_labels=["Q=5", "Q=10", "Q=15"],
        num_bins=2,
    )


class TestBoxPlotConfig:
    """Test BoxPlotConfig dataclass and factory function."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = BoxPlotConfig()
        assert config.x_label == "Uncertainty Thresholded Bin"
        assert config.y_label == "Error (%)"
        assert config.colormap == "Set1"
        assert config.show is False

    def test_create_boxplot_config_basic(self):
        """Test basic factory function usage."""
        config = create_boxplot_config(
            x_label="Custom X",
            y_label="Custom Y",
            colormap="Set2",
        )
        assert config.x_label == "Custom X"
        assert config.y_label == "Custom Y"
        assert config.colormap == "Set2"

    def test_create_boxplot_config_kwargs_filtering(self):
        """Test that invalid kwargs are filtered out."""
        config = create_boxplot_config(
            x_label="Valid",
            invalid_param="Should be filtered",
            another_invalid=123,
        )
        assert config.x_label == "Valid"
        # Should not raise error for invalid params


class TestBoxPlotData:
    """Test BoxPlotData dataclass and factory function."""

    def test_basic_data_creation(self, sample_evaluation_data):
        """Test basic data container creation."""
        data = create_boxplot_data(
            evaluation_data_by_bins=[sample_evaluation_data],
            uncertainty_categories=[["epistemic"]],
            models=["ResNet50"],
        )
        assert len(data.evaluation_data_by_bins) == 1
        assert data.uncertainty_categories == [["epistemic"]]
        assert data.models == ["ResNet50"]

    def test_data_creation_with_kwargs_filtering(self, sample_evaluation_data):
        """Test that invalid kwargs are filtered out."""
        data = create_boxplot_data(
            evaluation_data_by_bins=[sample_evaluation_data],
            uncertainty_categories=[["epistemic"]],
            models=["ResNet50"],
            invalid_param="Should be filtered",
        )
        assert len(data.evaluation_data_by_bins) == 1
        # Should not raise error for invalid params


class TestSampleInfoMode:
    """Test SampleInfoMode enumeration."""

    def test_enum_values(self):
        """Test enumeration values."""
        assert SampleInfoMode.NONE.value == "None"
        assert SampleInfoMode.ALL.value == "All"
        assert SampleInfoMode.AVERAGE.value == "Average"


class TestGenericBoxPlotter:
    """Test GenericBoxPlotter class."""

    def test_initialization(self, sample_generic_data, sample_config):
        """Test plotter initialization."""
        plotter = GenericBoxPlotter(sample_generic_data, sample_config)
        assert plotter.data == sample_generic_data
        assert plotter.config == sample_config
        assert plotter.processed_data is None

    def test_initialization_without_data(self, sample_config):
        """Test plotter initialization without data."""
        plotter = GenericBoxPlotter(config=sample_config)
        assert plotter.data is None
        assert plotter.config == sample_config

    def test_set_data(self, sample_generic_data, sample_config):
        """Test setting data after initialization."""
        plotter = GenericBoxPlotter(config=sample_config)
        plotter.set_data(sample_generic_data)
        assert plotter.data == sample_generic_data

    def test_process_data_without_data_raises_error(self, sample_config):
        """Test that processing data without setting it raises an error."""
        plotter = GenericBoxPlotter(config=sample_config)
        with pytest.raises(ValueError, match="BoxPlotData must be set before processing"):
            plotter.process_data()

    def test_process_data_missing_required_fields(self, sample_config):
        """Test that processing data with missing fields raises an error."""
        incomplete_data = BoxPlotData(evaluation_data_by_bins=[{}])
        plotter = GenericBoxPlotter(incomplete_data, sample_config)
        with pytest.raises(ValueError, match="GenericBoxPlotter requires"):
            plotter.process_data()

    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    def test_setup_plot(self, mock_gca, mock_style, sample_generic_data, sample_config):
        """Test plot setup functionality."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        plotter = GenericBoxPlotter(sample_generic_data, sample_config)
        plotter.setup_plot()

        mock_style.assert_called_once_with(sample_config.matplotlib_style)
        mock_ax.xaxis.grid.assert_called_once_with(False)
        assert plotter.legend_patches == []
        assert plotter.max_bin_height == 0.0

    @patch("kale.interpret.box_plot.save_or_show_plot")
    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.yticks")
    def test_draw_boxplot_basic(
        self,
        mock_yticks,
        mock_xticks,
        mock_subplots,
        mock_gca,
        mock_style,
        mock_save_show,
        sample_generic_data,
        sample_config,
    ):
        """Test basic boxplot drawing functionality."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Mock the boxplot return value to avoid empty sequence errors
        mock_boxplot_return = {
            "boxes": [MagicMock()],
            "medians": [MagicMock()],
            "means": [MagicMock()],
            "whiskers": [MagicMock(), MagicMock()],
            "caps": [MagicMock(), MagicMock()],
        }
        # Mock the y-data for caps to return some values
        mock_boxplot_return["caps"][-1].get_ydata.return_value = [10.0, 20.0]
        mock_ax.boxplot.return_value = mock_boxplot_return

        plotter = GenericBoxPlotter(sample_generic_data, sample_config)
        plotter.draw_boxplot()

        # Verify that the plot was set up
        mock_style.assert_called()
        mock_save_show.assert_called_once()


class TestPerModelBoxPlotter:
    """Test PerModelBoxPlotter class."""

    def test_initialization(self, sample_per_model_data, sample_config):
        """Test plotter initialization."""
        plotter = PerModelBoxPlotter(sample_per_model_data, sample_config)
        assert plotter.data == sample_per_model_data
        assert plotter.config == sample_config
        assert plotter.processed_data is None

    def test_process_data_without_data_raises_error(self, sample_config):
        """Test that processing data without setting it raises an error."""
        plotter = PerModelBoxPlotter(config=sample_config)
        with pytest.raises(ValueError, match="BoxPlotData must be set before processing"):
            plotter.process_data()

    def test_process_data_missing_required_fields(self, sample_config):
        """Test that processing data with missing fields raises an error."""
        incomplete_data = BoxPlotData(evaluation_data_by_bins=[{}])
        plotter = PerModelBoxPlotter(incomplete_data, sample_config)
        with pytest.raises(ValueError, match="PerModelBoxPlotter requires"):
            plotter.process_data()

    @patch("kale.interpret.box_plot.save_or_show_plot")
    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.yticks")
    def test_draw_boxplot_basic(
        self,
        mock_yticks,
        mock_xticks,
        mock_subplots,
        mock_gca,
        mock_style,
        mock_save_show,
        sample_per_model_data,
        sample_config,
    ):
        """Test basic boxplot drawing functionality."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Mock the boxplot return value to avoid empty sequence errors
        mock_boxplot_return = {
            "boxes": [MagicMock()],
            "medians": [MagicMock()],
            "means": [MagicMock()],
            "whiskers": [MagicMock(), MagicMock()],
            "caps": [MagicMock(), MagicMock()],
        }
        # Mock the y-data for caps to return some values
        mock_boxplot_return["caps"][-1].get_ydata.return_value = [10.0, 20.0]
        mock_ax.boxplot.return_value = mock_boxplot_return

        plotter = PerModelBoxPlotter(sample_per_model_data, sample_config)
        plotter.draw_boxplot()

        # Verify that the plot was set up
        mock_style.assert_called()
        mock_save_show.assert_called_once()


class TestComparingQBoxPlotter:
    """Test ComparingQBoxPlotter class."""

    def test_initialization(self, sample_comparing_q_data, sample_config):
        """Test plotter initialization."""
        plotter = ComparingQBoxPlotter(sample_comparing_q_data, sample_config)
        assert plotter.data == sample_comparing_q_data
        assert plotter.config == sample_config
        assert plotter.processed_data is None

    def test_process_data_without_data_raises_error(self, sample_config):
        """Test that processing data without setting it raises an error."""
        plotter = ComparingQBoxPlotter(config=sample_config)
        with pytest.raises(ValueError, match="BoxPlotData must be set before processing"):
            plotter.process_data()

    def test_process_data_missing_required_fields(self, sample_config):
        """Test that processing data with missing fields raises an error."""
        incomplete_data = BoxPlotData(evaluation_data_by_bins=[{}])
        plotter = ComparingQBoxPlotter(incomplete_data, sample_config)
        with pytest.raises(ValueError, match="For comparing_q plots"):
            plotter.process_data()

    @patch("kale.interpret.box_plot.save_or_show_plot")
    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.subplots_adjust")
    @patch("matplotlib.pyplot.xticks")
    @patch("matplotlib.pyplot.yticks")
    def test_draw_boxplot_basic(
        self,
        mock_yticks,
        mock_xticks,
        mock_subplots,
        mock_gca,
        mock_style,
        mock_save_show,
        sample_comparing_q_data,
        sample_config,
    ):
        """Test basic boxplot drawing functionality."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Mock the boxplot return value to avoid empty sequence errors
        mock_boxplot_return = {
            "boxes": [MagicMock()],
            "medians": [MagicMock()],
            "means": [MagicMock()],
            "whiskers": [MagicMock(), MagicMock()],
            "caps": [MagicMock(), MagicMock()],
        }
        # Mock the y-data for caps to return some values
        mock_boxplot_return["caps"][-1].get_ydata.return_value = [10.0, 20.0]
        mock_ax.boxplot.return_value = mock_boxplot_return

        # Need to set hatch_type for comparing Q plots
        sample_config.hatch_type = "///"
        plotter = ComparingQBoxPlotter(sample_comparing_q_data, sample_config)
        plotter.draw_boxplot()

        # Verify that the plot was set up
        mock_style.assert_called()
        mock_save_show.assert_called_once()


class TestHelperFunctions:
    """Test helper functions."""

    def test_find_data_key_success(self, sample_evaluation_data):
        """Test successful data key finding."""
        key = _find_data_key(sample_evaluation_data, "ResNet50", "epistemic")
        assert key == "ResNet50_epistemic"

    def test_find_data_key_failure(self, sample_evaluation_data):
        """Test data key finding failure."""
        with pytest.raises(KeyError, match="No matching key found"):
            _find_data_key(sample_evaluation_data, "NonExistent", "epistemic")

    def test_extract_bin_data_basic(self):
        """Test basic bin data extraction."""
        model_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = _extract_bin_data(model_data, 1, use_list_comp=False)
        assert result == [4, 5, 6]

    def test_extract_bin_data_with_none_filtering(self):
        """Test bin data extraction with None filtering."""
        model_data = [[1, None, 3], [4, 5, None], [7, 8, 9]]
        result = _extract_bin_data(model_data, 0, use_list_comp=True)
        assert result == [1, 3]

    def test_create_data_item(self):
        """Test data item creation."""
        model_data = [[1, 2, 3], [4, 5, 6]]
        item = _create_data_item(
            data=[1, 2, 3],
            x_position=1.0,
            width=0.2,
            color_idx=0,
            model_type="ResNet50",
            uncertainty_type="epistemic",
            hatch_idx=0,
            bin_idx=0,
            model_data=model_data,
            extra_field="extra_value",
        )

        assert item["data"] == [1, 2, 3]
        assert item["x_position"] == 1.0
        assert item["width"] == 0.2
        assert item["color_idx"] == 0
        assert item["model_type"] == "ResNet50"
        assert item["uncertainty_type"] == "epistemic"
        assert item["hatch_idx"] == 0
        assert item["bin_idx"] == 0
        assert item["model_data"] == model_data
        assert item["extra_field"] == "extra_value"

    def test_calculate_spacing_adjustment(self, sample_config):
        """Test spacing adjustment calculation."""
        # Test large spacing, num_bins > threshold
        result = _calculate_spacing_adjustment(15, True, sample_config)
        assert result == sample_config.gap_large_for_large_spacing

        # Test large spacing, num_bins <= threshold
        result = _calculate_spacing_adjustment(5, True, sample_config)
        assert result == sample_config.gap_small_for_large_spacing

        # Test small spacing, num_bins > threshold
        result = _calculate_spacing_adjustment(15, False, sample_config)
        assert result == sample_config.gap_large_for_small_spacing

        # Test small spacing, num_bins <= threshold
        result = _calculate_spacing_adjustment(5, False, sample_config)
        assert result == sample_config.gap_small_for_small_spacing


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("kale.interpret.box_plot.save_or_show_plot")
    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    def test_complete_generic_workflow(self, mock_gca, mock_style, mock_save_show, sample_evaluation_data, tmp_path):
        """Test complete workflow for generic plotting."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Mock the boxplot return value to avoid empty sequence errors
        mock_boxplot_return = {
            "boxes": [MagicMock()],
            "medians": [MagicMock()],
            "means": [MagicMock()],
            "whiskers": [MagicMock(), MagicMock()],
            "caps": [MagicMock(), MagicMock()],
        }
        # Mock the y-data for caps to return some values
        mock_boxplot_return["caps"][-1].get_ydata.return_value = [10.0, 20.0]
        mock_ax.boxplot.return_value = mock_boxplot_return

        # Create data and config
        data = create_boxplot_data(
            evaluation_data_by_bins=[sample_evaluation_data],
            uncertainty_categories=[["epistemic"]],
            models=["ResNet50", "VGG16"],
            category_labels=["B1", "B2", "B3"],
            num_bins=3,
        )
        config = create_boxplot_config(
            save_path=str(tmp_path / "test_plot.png"),
            show=False,
        )

        # Create and run plotter
        plotter = GenericBoxPlotter(data, config)
        plotter.draw_boxplot()

        # Verify workflow completed
        mock_save_show.assert_called_once()
        assert mock_save_show.call_args[1]["save_path"] == str(tmp_path / "test_plot.png")

    @patch("kale.interpret.box_plot.save_or_show_plot")
    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.gca")
    def test_complete_comparing_q_workflow(
        self, mock_gca, mock_style, mock_save_show, sample_q_comparison_data, tmp_path
    ):
        """Test complete workflow for Q-value comparison plotting."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Mock the boxplot return value to avoid empty sequence errors
        mock_boxplot_return = {
            "boxes": [MagicMock()],
            "medians": [MagicMock()],
            "means": [MagicMock()],
            "whiskers": [MagicMock(), MagicMock()],
            "caps": [MagicMock(), MagicMock()],
        }
        # Mock the y-data for caps to return some values
        mock_boxplot_return["caps"][-1].get_ydata.return_value = [10.0, 20.0]
        mock_ax.boxplot.return_value = mock_boxplot_return

        # Create data and config
        data = create_boxplot_data(
            evaluation_data_by_bins=sample_q_comparison_data,
            uncertainty_categories=[["epistemic"]],
            models=["model"],
            category_labels=["Q=5", "Q=10", "Q=15"],
            num_bins=2,
        )
        config = create_boxplot_config(
            hatch_type="///",
            save_path=str(tmp_path / "test_q_plot.png"),
            show=False,
        )

        # Create and run plotter
        plotter = ComparingQBoxPlotter(data, config)
        plotter.draw_boxplot()

        # Verify workflow completed
        mock_save_show.assert_called_once()
        assert mock_save_show.call_args[1]["save_path"] == str(tmp_path / "test_q_plot.png")

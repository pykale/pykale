from unittest.mock import MagicMock, patch

import pytest

# Import only the core components to avoid dependency issues
try:
    from kale.interpret.box_plot import (
        BoxPlotConfig,
        BoxPlotData,
        BoxPlotDataProcessor,
        ComparingQBoxPlotDataProcessor,
        ComparingQBoxPlotter,
        create_boxplot_config,
        create_boxplot_data,
        GenericBoxPlotDataProcessor,
        GenericBoxPlotter,
        PerModelBoxPlotDataProcessor,
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


class TestBoxPlotDataProcessor:
    """Test BoxPlotDataProcessor methods and subclasses."""

    @pytest.fixture
    def processor(self):
        """Create a generic BoxPlotDataProcessor instance for testing."""
        config = create_boxplot_config()
        return GenericBoxPlotDataProcessor(config)

    @pytest.fixture
    def processor_per_model(self):
        """Create a PerModelBoxPlotDataProcessor instance for testing."""
        config = create_boxplot_config()
        return PerModelBoxPlotDataProcessor(config)

    @pytest.fixture
    def processor_comparing_q(self):
        """Create a ComparingQBoxPlotDataProcessor instance for testing."""
        config = create_boxplot_config()
        return ComparingQBoxPlotDataProcessor(config)

    def test_processor_initialization(self, processor):
        """Test that processors are initialized correctly."""
        # Check that all attributes are initialized
        assert processor.config is not None
        assert processor.evaluation_data_by_bin is None
        assert processor.uncertainty_categories is None
        assert processor.models is None
        assert processor.num_bins is None
        assert processor.outer_min_x_loc == 0.0
        assert processor.middle_min_x_loc == 0.0
        assert processor.inner_min_x_loc == 0.0
        assert processor.processed_data == []
        assert processor.legend_info == []
        assert processor.bin_label_locs == []

    def test_find_data_key_success(self, processor, sample_evaluation_data):
        """Test successful data key finding."""
        key = processor._find_data_key(sample_evaluation_data, "ResNet50", "epistemic")
        assert key == "ResNet50_epistemic"

    def test_find_data_key_failure(self, processor, sample_evaluation_data):
        """Test data key finding failure."""
        with pytest.raises(KeyError, match="No matching key found"):
            processor._find_data_key(sample_evaluation_data, "NonExistent", "epistemic")

    def test_extract_bin_data_basic(self, processor):
        """Test basic bin data extraction."""
        model_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = processor._extract_bin_data(model_data, 1)
        assert result == [4, 5, 6]

    def test_extract_bin_data_with_none_filtering(self, processor):
        """Test bin data extraction with None filtering."""
        processor.config.use_list_comp = True
        model_data = [[1, None, 2], [3, 4, None], [5, 6, 7]]

        result = processor._extract_bin_data(model_data, 0)
        assert result == [1, 2]  # None values filtered out

        result = processor._extract_bin_data(model_data, 1)
        assert result == [3, 4]  # None values filtered out

    def test_extract_bin_data_without_none_filtering(self, processor):
        """Test bin data extraction without None filtering."""
        processor.config.use_list_comp = False
        model_data = [[1, None, 2], [3, 4, None], [5, 6, 7]]

        result = processor._extract_bin_data(model_data, 0)
        assert result == [1, None, 2]  # None values preserved

    def test_calculate_spacing_adjustment_large(self, processor):
        """Test spacing adjustment calculation for large spacing."""
        processor.num_bins = 10  # Above large threshold
        result = processor._calculate_spacing_adjustment(is_large_spacing=True)
        expected = processor.config.gap_large_for_large_spacing
        assert result == expected

    def test_calculate_spacing_adjustment_small(self, processor):
        """Test spacing adjustment calculation for small spacing."""
        processor.num_bins = 2  # Below threshold
        result = processor._calculate_spacing_adjustment(is_large_spacing=True)
        expected = processor.config.gap_small_for_large_spacing
        assert result == expected

    def test_create_data_item(self):
        """Test static data item creation method."""
        data = [0.1, 0.2, 0.3]
        model_data = [[0.1, 0.2], [0.3, 0.4]]

        item = BoxPlotDataProcessor.create_data_item(
            data=data,
            x_position=1.5,
            width=0.3,
            color_idx=0,
            model_type="ResNet50",
            uncertainty_type="epistemic",
            hatch_idx=1,
            bin_idx=0,
            model_data=model_data,
            extra_field="test_value",
        )

        assert item["data"] == data
        assert item["x_position"] == 1.5
        assert item["width"] == 0.3
        assert item["color_idx"] == 0
        assert item["model_type"] == "ResNet50"
        assert item["uncertainty_type"] == "epistemic"
        assert item["hatch_idx"] == 1
        assert item["bin_idx"] == 0
        assert item["model_data"] == model_data
        assert item["extra_field"] == "test_value"

    def test_process_single_item(self, processor, sample_evaluation_data):
        """Test processing a single data item."""
        # Set up processor state
        processor.evaluation_data_by_bin = sample_evaluation_data
        processor.config.use_list_comp = False
        processor.outer_min_x_loc = 1.0
        processor.middle_min_x_loc = 0.5
        processor.inner_min_x_loc = 0.2

        item = processor._process_single_item(
            uncertainty_type="epistemic", model_type="ResNet50", bin_idx=0, uncertainty_idx=0, hatch_idx=0, width=0.3
        )

        assert item["data"] == sample_evaluation_data["ResNet50_epistemic"][0]
        assert item["x_position"] == 1.7  # 1.0 + 0.5 + 0.2
        assert item["width"] == 0.3
        assert item["model_type"] == "ResNet50"
        assert item["uncertainty_type"] == "epistemic"

    def test_process_and_store_single_item(self, processor, sample_evaluation_data):
        """Test processing and storing a single item."""
        # Set up processor state
        processor.evaluation_data_by_bin = sample_evaluation_data
        processor.config.use_list_comp = False
        processor.outer_min_x_loc = 1.0
        processor.middle_min_x_loc = 0.5
        processor.inner_min_x_loc = 0.2

        x_position = processor._process_and_store_single_item(
            uncertainty_type="epistemic", model_type="ResNet50", bin_idx=0, uncertainty_idx=0, hatch_idx=0, width=0.3
        )

        assert x_position == 1.7  # 1.0 + 0.5 + 0.2
        assert len(processor.processed_data) == 1
        assert processor.processed_data[0]["model_type"] == "ResNet50"

    def test_collect_legend_info_for_first_bin(self, processor):
        """Test legend information collection for first bin."""
        processor.legend_info = []

        processor._collect_legend_info_for_first_bin(
            bin_idx=0, uncertainty_idx=1, model_type="VGG16", uncertainty_type="aleatoric", hatch_idx=2
        )

        assert len(processor.legend_info) == 1
        legend_item = processor.legend_info[0]
        assert legend_item["color_idx"] == 1
        assert legend_item["model_type"] == "VGG16"
        assert legend_item["uncertainty_type"] == "aleatoric"
        assert legend_item["hatch_idx"] == 2

    def test_collect_legend_info_skip_non_first_bin(self, processor):
        """Test that legend info is not collected for non-first bins."""
        processor.legend_info = []

        processor._collect_legend_info_for_first_bin(
            bin_idx=1, uncertainty_idx=1, model_type="VGG16", uncertainty_type="aleatoric", hatch_idx=2  # Not first bin
        )

        assert len(processor.legend_info) == 0

    def test_process_and_collect_positions(self, processor, sample_evaluation_data):
        """Test processing and collecting positions for multiple items."""
        # Set up processor state
        processor.evaluation_data_by_bin = sample_evaluation_data
        processor.config.use_list_comp = False
        processor.config.inner_spacing = 0.1
        processor.outer_min_x_loc = 1.0
        processor.middle_min_x_loc = 0.5
        processor.inner_min_x_loc = 0.0
        processor.legend_info = []

        items_data = [
            (0, "ResNet50", 0),  # (bin_idx, model_type, hatch_idx)
            (0, "VGG16", 1),
        ]

        box_x_positions = processor._process_and_collect_positions(
            uncertainty_type="epistemic", uncertainty_idx=0, items_data=items_data, width=0.3
        )

        assert len(box_x_positions) == 2
        assert box_x_positions[0] == 1.5  # 1.0 + 0.5 + 0.0
        # Second position should account for spacing and width updates
        assert len(processor.processed_data) == 2
        assert len(processor.legend_info) == 2  # Both items should have legend info (bin_idx=0)

    def test_store_bin_label_positions_extend(self, processor):
        """Test storing bin label positions with extend mode."""
        processor.bin_label_locs = [1.0, 2.0]
        box_x_positions = [3.0, 4.0, 5.0]

        processor._store_bin_label_positions(box_x_positions, use_extend=True)

        assert processor.bin_label_locs == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_store_bin_label_positions_mean(self, processor):
        """Test storing bin label positions with mean mode."""
        processor.bin_label_locs = [1.0, 2.0]
        box_x_positions = [3.0, 4.0, 5.0]

        processor._store_bin_label_positions(box_x_positions, use_extend=False)

        assert processor.bin_label_locs == [1.0, 2.0, 4.0]  # Mean of [3.0, 4.0, 5.0] is 4.0

    def test_generic_processor_process_data(self, processor, sample_evaluation_data):
        """Test GenericBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs = processor.process_data(
            sample_evaluation_data, [["epistemic"], ["aleatoric"]], ["ResNet50", "VGG16"], 3
        )

        # Should have processed data for all combinations
        # 2 uncertainties × 3 bins × 2 models = 12 items
        assert len(processed_data) == 12
        assert legend_info is not None
        assert len(legend_info) == 4  # 2 uncertainties × 2 models
        assert len(bin_label_locs) > 0

    def test_per_model_processor_process_data(self, processor_per_model, sample_evaluation_data):
        """Test PerModelBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs = processor_per_model.process_data(
            sample_evaluation_data, [["epistemic"], ["aleatoric"]], ["ResNet50", "VGG16"], 3
        )

        # Should have processed data for all combinations
        # 2 uncertainties × 2 models × 3 bins = 12 items
        assert len(processed_data) == 12
        assert legend_info is not None
        assert len(legend_info) == 4  # 2 uncertainties × 2 models
        assert len(bin_label_locs) > 0

    def test_comparing_q_processor_process_data(self, processor_comparing_q, sample_q_comparison_data):
        """Test ComparingQBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs = processor_comparing_q.process_data(
            sample_q_comparison_data, [["epistemic"]], ["model"], ["Q=5", "Q=10", "Q=15"]
        )

        # Should have processed data for all Q values and bins
        assert len(processed_data) == 6  # 3 Q values × 2 bins per Q value
        assert legend_info is None  # Q-comparison mode doesn't use legend_info
        assert len(bin_label_locs) == 3  # 3 Q values (each Q value gets one label position)

    def test_processor_empty_data_validation(self, processor):
        """Test that processors validate empty data correctly."""
        with pytest.raises(ValueError, match="Processing resulted in empty data"):
            # Try to finalize with empty processed_data
            processor._finalize_processing_results()

    def test_processor_empty_bin_labels_validation(self, processor):
        """Test that processors validate empty bin labels correctly."""
        processor.processed_data = [{"test": "data"}]  # Non-empty processed_data
        processor.bin_label_locs = []  # Empty bin labels

        with pytest.raises(ValueError, match="Processing resulted in empty bin label locations"):
            processor._finalize_processing_results()


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

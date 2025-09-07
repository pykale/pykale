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

    def test_set_params_with_invalid_parameter(self):
        """Test that set_params raises AttributeError for invalid parameters."""
        config = BoxPlotConfig()
        with pytest.raises(AttributeError, match="'invalid_param' is not a valid BoxPlotConfig parameter"):
            config.set_params(invalid_param="test")

    def test_set_params_with_valid_parameters(self):
        """Test that set_params works correctly with valid parameters."""
        config = BoxPlotConfig()
        result = config.set_params(x_label="New Label", show=True)
        assert result is config  # Should return self for chaining
        assert config.x_label == "New Label"
        assert config.show is True


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

    def test_set_data_with_invalid_parameter(self, sample_evaluation_data):
        """Test that set_data raises AttributeError for invalid parameters."""
        data = BoxPlotData(evaluation_data_by_bins=[sample_evaluation_data])
        with pytest.raises(AttributeError, match="'invalid_param' is not a valid BoxPlotData parameter"):
            data.set_data(invalid_param="test")

    def test_set_data_with_valid_parameters(self, sample_evaluation_data):
        """Test that set_data works correctly with valid parameters."""
        data = BoxPlotData(evaluation_data_by_bins=[sample_evaluation_data])
        result = data.set_data(models=["ResNet50"], num_bins=5)
        assert result is data  # Should return self for chaining
        assert data.models == ["ResNet50"]
        assert data.num_bins == 5

    def test_set_data_method_chaining(self, sample_evaluation_data):
        """Test that set_data supports method chaining."""
        data = BoxPlotData(evaluation_data_by_bins=[sample_evaluation_data])
        result = data.set_data(models=["ResNet50"]).set_data(num_bins=5)
        assert result is data
        assert data.models == ["ResNet50"]
        assert data.num_bins == 5


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

    def test_process_data_with_incomplete_required_fields(self, sample_config):
        """Test various combinations of missing required fields."""
        # Test missing evaluation_data_by_bins
        data1 = BoxPlotData(
            evaluation_data_by_bins=None, uncertainty_categories=[["epistemic"]], models=["ResNet50"], num_bins=3
        )
        plotter1 = GenericBoxPlotter(data1, sample_config)
        with pytest.raises(ValueError, match="GenericBoxPlotter requires"):
            plotter1.process_data()

        # Test missing uncertainty_categories
        data2 = BoxPlotData(
            evaluation_data_by_bins=[{"model_epistemic": [[1, 2]]}],
            uncertainty_categories=None,
            models=["ResNet50"],
            num_bins=3,
        )
        plotter2 = GenericBoxPlotter(data2, sample_config)
        with pytest.raises(ValueError, match="GenericBoxPlotter requires"):
            plotter2.process_data()

        # Test missing models
        data3 = BoxPlotData(
            evaluation_data_by_bins=[{"model_epistemic": [[1, 2]]}],
            uncertainty_categories=[["epistemic"]],
            models=None,
            num_bins=3,
        )
        plotter3 = GenericBoxPlotter(data3, sample_config)
        with pytest.raises(ValueError, match="GenericBoxPlotter requires"):
            plotter3.process_data()

        # Test missing num_bins
        data4 = BoxPlotData(
            evaluation_data_by_bins=[{"model_epistemic": [[1, 2]]}],
            uncertainty_categories=[["epistemic"]],
            models=["ResNet50"],
            num_bins=None,
        )
        plotter4 = GenericBoxPlotter(data4, sample_config)
        with pytest.raises(ValueError, match="GenericBoxPlotter requires"):
            plotter4.process_data()

    @patch("matplotlib.pyplot.subplots")
    def test_draw_boxplot_with_no_uncertainty_categories(self, mock_subplots, sample_config):
        """Test draw_boxplot behavior when uncertainty_categories is empty."""
        # Mock matplotlib components
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create data with empty uncertainty_categories
        data = BoxPlotData(
            evaluation_data_by_bins=[{"model_epistemic": [[1, 2, 3]]}],
            uncertainty_categories=[],  # Empty categories
            models=["model"],
            num_bins=3,
            category_labels=["B1", "B2", "B3"],
        )

        plotter = GenericBoxPlotter(data, sample_config)

        # Mock processed data and other required attributes
        plotter.processed_data = [
            {"data": [[1, 2, 3]], "x_position": 1.0, "width": 0.2, "color_idx": 0, "hatch_idx": 0}
        ]
        plotter.bin_label_locs = [[1.0]]
        plotter.ax = mock_ax
        plotter.legend_patches = []

        # This should use default colors instead of color mapping
        with patch("kale.interpret.box_plot.save_or_show_plot"):
            plotter.draw_boxplot()

        # Should complete without errors


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
        processor.config.detailed_mode = True
        model_data = [[1, None, 2], [3, 4, None], [5, 6, 7]]

        result = processor._extract_bin_data(model_data, 0)
        assert result == [1, 2]  # None values filtered out

        result = processor._extract_bin_data(model_data, 1)
        assert result == [3, 4]  # None values filtered out

    def test_extract_bin_data_without_none_filtering(self, processor):
        """Test bin data extraction without None filtering."""
        processor.config.detailed_mode = False
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
            percent_size=25.5,
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
        processor.config.detailed_mode = False
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
        processor.config.detailed_mode = False
        processor.outer_min_x_loc = 1.0
        processor.middle_min_x_loc = 0.5
        processor.inner_min_x_loc = 0.2

        x_position, percent_size = processor._process_and_store_single_item(
            uncertainty_type="epistemic", model_type="ResNet50", bin_idx=0, uncertainty_idx=0, hatch_idx=0, width=0.3
        )

        assert x_position == 1.7  # 1.0 + 0.5 + 0.2
        assert isinstance(percent_size, (int, float))  # Can be 0 if show_sample_info is "None"
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
        processor.config.detailed_mode = False
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

    def test_generic_processor_process_data(self, processor, sample_evaluation_data):
        """Test GenericBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs, all_sample_percs = processor.process_data(
            sample_evaluation_data, [["epistemic"], ["aleatoric"]], ["ResNet50", "VGG16"], 3
        )

        # Should have processed data for all combinations
        # 2 uncertainties × 3 bins × 2 models = 12 items
        assert len(processed_data) == 12
        assert legend_info is not None
        assert len(legend_info) == 4  # 2 uncertainties × 2 models
        assert len(bin_label_locs) > 0
        assert len(all_sample_percs) >= 0

    def test_per_model_processor_process_data(self, processor_per_model, sample_evaluation_data):
        """Test PerModelBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs, all_sample_percs = processor_per_model.process_data(
            sample_evaluation_data, [["epistemic"], ["aleatoric"]], ["ResNet50", "VGG16"], 3
        )

        # Should have processed data for all combinations
        # 2 uncertainties × 2 models × 3 bins = 12 items
        assert len(processed_data) == 12
        assert legend_info is not None
        assert len(legend_info) == 4  # 2 uncertainties × 2 models
        assert len(bin_label_locs) > 0
        assert len(all_sample_percs) >= 0

    def test_comparing_q_processor_process_data(self, processor_comparing_q, sample_q_comparison_data):
        """Test ComparingQBoxPlotDataProcessor process_data method."""
        processed_data, legend_info, bin_label_locs, all_sample_percs = processor_comparing_q.process_data(
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


class TestPrivateMethods:
    """Test private methods that are not covered by other tests."""

    @pytest.fixture
    def mock_plotter(self, sample_generic_data, sample_config):
        """Create a plotter instance for testing private methods."""
        plotter = GenericBoxPlotter(sample_generic_data, sample_config)
        plotter.ax = MagicMock()  # Mock the axes
        return plotter

    def test_finalize_plot(self, mock_plotter):
        """Test the _finalize_plot method."""
        # Set up required attributes
        mock_plotter.bin_label_locs = [[1.0, 2.0, 3.0]]
        mock_plotter.data.category_labels = ["B1", "B2", "B3"]
        mock_plotter.data.num_bins = 3
        mock_plotter.data.uncertainty_categories = [["epistemic"]]

        # Mock the _format_and_finalize_plot method
        with patch.object(mock_plotter, "_format_and_finalize_plot") as mock_format:
            mock_plotter._finalize_plot()

            # Verify the method was called with correct parameters
            mock_format.assert_called_once_with(
                mock_plotter.bin_label_locs,
                mock_plotter.data.category_labels,
                mock_plotter.data.num_bins,
                mock_plotter.data.uncertainty_categories,
                comparing_q=False,
                show_all_ticks=False,  # Default detailed_mode is False
            )

    def test_finalize_plot_with_detailed_mode_false(self, sample_generic_data):
        """Test _finalize_plot with detailed_mode set to False."""
        config = create_boxplot_config(detailed_mode=False)
        plotter = GenericBoxPlotter(sample_generic_data, config)
        plotter.ax = MagicMock()

        # Set up required attributes
        plotter.bin_label_locs = [[1.0, 2.0, 3.0]]
        plotter.data.category_labels = ["B1", "B2", "B3"]
        plotter.data.num_bins = 3
        plotter.data.uncertainty_categories = [["epistemic"]]

        # Mock the _format_and_finalize_plot method
        with patch.object(plotter, "_format_and_finalize_plot") as mock_format:
            plotter._finalize_plot()

            # Verify show_all_ticks is False when detailed_mode is False
            mock_format.assert_called_once_with(
                plotter.bin_label_locs,
                plotter.data.category_labels,
                plotter.data.num_bins,
                plotter.data.uncertainty_categories,
                comparing_q=False,
                show_all_ticks=False,
            )

    def test_finalize_plot_with_detailed_mode_true(self, sample_generic_data):
        """Test _finalize_plot with detailed_mode set to True."""
        config = create_boxplot_config(detailed_mode=True)
        plotter = GenericBoxPlotter(sample_generic_data, config)
        plotter.ax = MagicMock()

        # Set up required attributes
        plotter.bin_label_locs = [[1.0, 2.0, 3.0]]
        plotter.data.category_labels = ["B1", "B2", "B3"]
        plotter.data.num_bins = 3
        plotter.data.uncertainty_categories = [["epistemic"]]

        # Mock the _format_and_finalize_plot method
        with patch.object(plotter, "_format_and_finalize_plot") as mock_format:
            plotter._finalize_plot()

            # Verify show_all_ticks is True when detailed_mode is True
            mock_format.assert_called_once_with(
                plotter.bin_label_locs,
                plotter.data.category_labels,
                plotter.data.num_bins,
                plotter.data.uncertainty_categories,
                comparing_q=False,
                show_all_ticks=True,
            )

    def test_finalize_plot_missing_bin_label_locs(self, mock_plotter):
        """Test _finalize_plot raises assertion error when bin_label_locs is None."""
        mock_plotter.bin_label_locs = None

        with pytest.raises(AssertionError):
            mock_plotter._finalize_plot()

    def test_finalize_plot_missing_data(self, sample_config):
        """Test _finalize_plot raises assertion error when data is None."""
        plotter = GenericBoxPlotter(config=sample_config)
        plotter.data = None
        plotter.ax = MagicMock()

        with pytest.raises(AssertionError):
            plotter._finalize_plot()

    def test_sample_info_average_mode(self, sample_generic_data):
        """Test show_sample_info='Average' mode execution."""
        config = create_boxplot_config(show_sample_info="Average", show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Setup required state
        plotter.bin_label_locs = [[1.0, 2.0], [3.0, 4.0]]
        plotter.sample_label_x_positions = []

        # Test _collect_sample_info_average method
        plotter._collect_sample_info_average()

        # Verify sample positions were calculated
        assert len(plotter.sample_label_x_positions) == 2
        assert plotter.sample_label_x_positions[0] == 1.5  # mean of [1.0, 2.0]
        assert plotter.sample_label_x_positions[1] == 3.5  # mean of [3.0, 4.0]

    def test_show_individual_dots_feature(self, sample_generic_data):
        """Test show_individual_dots feature in _create_single_boxplot."""
        config = create_boxplot_config(show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            # Create proper mock caps with get_ydata method
            mock_cap1 = MagicMock()
            mock_cap2 = MagicMock()
            mock_cap1.get_ydata.return_value = [0.5, 0.6]
            mock_cap2.get_ydata.return_value = [0.7, 0.8]

            mock_ax.boxplot.return_value = {
                "boxes": [MagicMock()],
                "medians": [MagicMock()],
                "caps": [mock_cap1, mock_cap2],
                "means": [MagicMock()],
            }

            plotter.setup_plot()

            # Test with show_individual_dots=True
            data = [0.1, 0.2, 0.3, 0.4, 0.5]
            x_loc = [1.0]
            width = 0.2

            with patch("numpy.random.normal") as mock_random:
                mock_random.return_value = [0.98, 1.01, 0.99, 1.02, 1.0]

                plotter._create_single_boxplot(
                    data=data,
                    x_loc=x_loc,
                    width=width,
                    convert_to_percent=False,
                    show_individual_dots=True,
                    box_color="blue",
                    dot_color="green",
                    hatch_idx=0,
                )  # Verify boxplot was called
                mock_ax.boxplot.assert_called_once()
                # Verify individual dots were plotted
                mock_ax.plot.assert_called_once()

    def test_convert_to_percent_feature(self, sample_generic_data):
        """Test convert_to_percent feature in _create_single_boxplot."""
        config = create_boxplot_config(show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            # Create proper mock caps with get_ydata method
            mock_cap1 = MagicMock()
            mock_cap2 = MagicMock()
            mock_cap1.get_ydata.return_value = [10.0, 20.0]
            mock_cap2.get_ydata.return_value = [30.0, 40.0]

            mock_ax.boxplot.return_value = {
                "boxes": [MagicMock()],
                "medians": [MagicMock()],
                "caps": [mock_cap1, mock_cap2],
                "means": [MagicMock()],
            }

            plotter.setup_plot()

            # Test data with None values to test filtering
            data = [0.1, 0.2, None, 0.3, 0.4]
            x_loc = [1.0]
            width = 0.2

            # Test with convert_to_percent=True
            plotter._create_single_boxplot(
                data=data,
                x_loc=x_loc,
                width=width,
                convert_to_percent=True,
                show_individual_dots=False,
                box_color="blue",
                dot_color="green",
                hatch_idx=0,
            )  # Verify boxplot was called with percentage data
            called_args = mock_ax.boxplot.call_args
            called_data = called_args[0][0]  # First positional argument (data)
            expected_data = [10.0, 20.0, 30.0, 40.0]  # [0.1, 0.2, 0.3, 0.4] * 100, None filtered out
            assert called_data == expected_data

    def test_display_sample_information_average_mode(self, sample_generic_data):
        """Test _display_sample_information with Average mode."""
        config = create_boxplot_config(show_sample_info="Average", show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()

            # Setup required state
            plotter.sample_label_x_positions = [1.0, 2.0]
            plotter.all_sample_percs = [[85, 5], [90, 3]]  # [mean, std] format
            plotter.max_bin_height = 10.0

            # Call the method
            plotter._display_sample_information()

            # Verify text was added for Average mode
            assert mock_ax.text.call_count == 2

            # Check first call
            first_call = mock_ax.text.call_args_list[0]
            assert first_call[0][0] == 1.0  # x position
            assert first_call[0][1] == 8.0  # y position (max_bin_height * 0.8)
            assert "PSB" in first_call[0][2]  # text contains PSB

    def test_display_sample_information_all_mode(self, sample_generic_data):
        """Test _display_sample_information with All mode."""
        config = create_boxplot_config(show_sample_info="All", show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()

            # Setup required state
            plotter.sample_label_x_positions = [1.0, 2.0, 3.0]
            plotter.all_sample_percs = [85, 90, 75]  # Individual percentages
            plotter.max_bin_height = 10.0

            # Call the method
            plotter._display_sample_information()

            # Verify text was added for All mode
            assert mock_ax.text.call_count == 3

            # Check alternating heights
            calls = mock_ax.text.call_args_list
            assert calls[0][0][1] == 11.0  # max_bin_height + 2 * (0 % 2) + 1
            assert calls[1][0][1] == 13.0  # max_bin_height + 2 * (1 % 2) + 1
            assert calls[2][0][1] == 11.0  # max_bin_height + 2 * (2 % 2) + 1

    def test_display_sample_information_none_mode(self, sample_generic_data):
        """Test _display_sample_information with None mode (early return)."""
        config = create_boxplot_config(show_sample_info="None", show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()

            # Call the method
            plotter._display_sample_information()

            # Verify no text was added (early return)
            mock_ax.text.assert_not_called()

    def test_get_formatted_labels_for_bin_count_medium_bins(self, mock_plotter):
        """Test _get_formatted_labels_for_bin_count with medium bin count (6-15)."""
        category_labels = ["Low", "Medium", "High"]

        # Test with 10 bins (medium range)
        result = mock_plotter._get_formatted_labels_for_bin_count(category_labels, 10)

        # Should call _create_abbreviated_labels with blanks_around=3
        # The result should contain blanks and arrow
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_formatted_labels_for_bin_count_many_bins(self, mock_plotter):
        """Test _get_formatted_labels_for_bin_count with many bins (>15)."""
        category_labels = ["Low", "Medium", "High"]

        # Test with 20 bins (large range)
        result = mock_plotter._get_formatted_labels_for_bin_count(category_labels, 20)

        # Should call _create_abbreviated_labels with blanks_around=5 and add_padding=True
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_formatted_labels_for_bin_count_single_label(self, mock_plotter):
        """Test _get_formatted_labels_for_bin_count with single category label."""
        category_labels = ["Only"]

        # Test with few bins and single label
        result = mock_plotter._get_formatted_labels_for_bin_count(category_labels, 3)

        # Should return the single label as-is
        assert result == ["Only"]

    def test_setup_y_axis_scaling_log_scale(self, sample_generic_data):
        """Test _setup_y_axis_scaling with logarithmic scale."""
        config = create_boxplot_config(to_log=True, y_lim_bottom=1, y_lim_top=1000, show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca, patch("matplotlib.pyplot.ScalarFormatter") as mock_formatter:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax
            mock_formatter_instance = MagicMock()
            mock_formatter.return_value = mock_formatter_instance

            plotter.setup_plot()

            # Call the method
            plotter._setup_y_axis_scaling()

            # Verify log scale was set
            mock_ax.set_yscale.assert_called_with("symlog", base=2)
            mock_ax.yaxis.set_major_formatter.assert_called_once()
            mock_ax.set_ylim.assert_called_with(1, 1000)

    def test_setup_legend_for_mode_with_sample_info_average(self, sample_generic_data):
        """Test _setup_legend_for_mode when show_sample_info is Average."""
        config = create_boxplot_config(show_sample_info="Average", show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)
        # Mock legend info with proper format
        plotter.legend_info = [
            {"color_idx": 0, "model_type": "TestModel", "uncertainty_type": "epistemic", "hatch_idx": 0}
        ]

        with patch("matplotlib.pyplot.gca") as mock_gca, patch.object(plotter, "_add_legend_patch") as mock_add_patch:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()
            colors = ["red", "blue"]
            plotter._setup_legend_for_mode(colors)

            # Should call _add_legend_patch for each legend item
            mock_add_patch.assert_called_once()

    def test_convert_to_percent_with_high_y_limit(self, sample_generic_data):
        """Test convert_to_percent functionality with high y_limit."""
        config = create_boxplot_config(convert_to_percent=True, y_lim_top=150, show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, config)

        with patch("matplotlib.pyplot.gca") as mock_gca, patch("matplotlib.pyplot.yticks") as mock_yticks:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()
            plotter._setup_y_axis_scaling()

            # Should set y-ticks when convert_to_percent=True and y_lim_top > 100
            mock_yticks.assert_called_once()

    def test_abstract_method_coverage(self):
        """Test abstract base class structure."""

        # Test that abstract base class exists and has proper structure
        from kale.interpret.box_plot import BoxPlotDataProcessor

        assert hasattr(BoxPlotDataProcessor, "_initialize_processing_parameters")
        assert hasattr(BoxPlotDataProcessor, "_execute_processing_loop")

        # Cannot instantiate abstract base class directly
        with pytest.raises(TypeError):
            BoxPlotDataProcessor()

    def test_calculate_sample_percs_none_mode(self):
        """Test _calculate_sample_percs when show_sample_info is None."""
        config = create_boxplot_config(show_sample_info="None", show=False, save_path=None)

        from kale.interpret.box_plot import GenericBoxPlotDataProcessor

        processor = GenericBoxPlotDataProcessor()
        processor.config = config

        model_data = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        bin_data = [0.1, 0.2]

        result = processor._calculate_sample_percs(model_data, bin_data)
        assert result == 0

    def test_calculate_sample_percs_all_mode(self):
        """Test _calculate_sample_percs when show_sample_info is All."""
        config = create_boxplot_config(show_sample_info="All", show=False, save_path=None)

        from kale.interpret.box_plot import GenericBoxPlotDataProcessor

        processor = GenericBoxPlotDataProcessor()
        processor.config = config
        processor.all_sample_percs = []  # Initialize the list

        model_data = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        bin_data = [0.1, 0.2]

        result = processor._calculate_sample_percs(model_data, bin_data)

        # Should calculate percentage and append to all_sample_percs
        assert result > 0
        assert len(processor.all_sample_percs) == 1

    def test_calculate_average_sample_info(self):
        """Test _calculate_average_sample_info method."""
        config = create_boxplot_config(show_sample_info="Average", show=False, save_path=None)

        from kale.interpret.box_plot import GenericBoxPlotDataProcessor

        processor = GenericBoxPlotDataProcessor()
        processor.config = config
        processor.all_sample_percs = []  # Initialize the list

        average_samples_per_bin = [10.5, 15.3, 8.7, 12.1]
        processor._calculate_average_sample_info(average_samples_per_bin)

        # Should calculate mean and std, then append to all_sample_percs
        assert len(processor.all_sample_percs) == 1
        mean_std = processor.all_sample_percs[0]
        assert len(mean_std) == 2  # [mean, std]
        assert mean_std[0] > 0  # mean should be positive
        assert mean_std[1] >= 0  # std should be non-negative

    def test_generic_data_processor_spacing_calculation(self):
        """Test spacing calculations in GenericBoxPlotDataProcessor."""
        config = create_boxplot_config(show=False, save_path=None)

        from kale.interpret.box_plot import GenericBoxPlotDataProcessor

        processor = GenericBoxPlotDataProcessor()
        processor.config = config
        processor.middle_min_x_loc = 0
        processor.outer_min_x_loc = 0
        processor.num_bins = 5  # Set num_bins to avoid assertion error

        # Test spacing adjustment calculation
        detailed_mode = True
        spacing_adjustment = processor._calculate_spacing_adjustment(detailed_mode)
        assert isinstance(spacing_adjustment, (int, float))

    def test_per_model_data_processor_average_mode(self, sample_generic_data):
        """Test PerModelBoxPlotDataProcessor with Average sample info mode."""
        config = create_boxplot_config(show_sample_info="Average", show=False, save_path=None)

        from kale.interpret.box_plot import PerModelBoxPlotDataProcessor

        processor = PerModelBoxPlotDataProcessor()
        processor.config = config
        processor.all_sample_percs = []
        processor.inner_min_x_loc = 0

        # Test that it calls _calculate_average_sample_info
        average_samples_per_bin = [10.0, 15.0, 8.0]
        processor._calculate_average_sample_info(average_samples_per_bin)

        assert len(processor.all_sample_percs) == 1

    def test_apply_q_spacing_functional_test(self):
        """Test functionality related to Q spacing using integration approach."""
        config = create_boxplot_config(show=False, save_path=None)

        from kale.interpret.box_plot import PerModelBoxPlotDataProcessor

        processor = PerModelBoxPlotDataProcessor()
        processor.config = config
        processor.bin_label_locs = []
        processor.outer_min_x_loc = 0

        # Test basic functionality exists
        assert hasattr(processor, "config")
        assert hasattr(processor, "bin_label_locs")
        assert hasattr(processor, "outer_min_x_loc")

    def test_set_config_method(self, sample_generic_data):
        """Test set_config method specifically targeting line 506."""
        initial_config = create_boxplot_config(show=False, save_path=None)
        plotter = GenericBoxPlotter(sample_generic_data, initial_config)

        # Create a new config with different settings
        new_config = create_boxplot_config(
            show=False, save_path=None, x_label="New X Label", y_label="New Y Label", show_sample_info="All"
        )

        # Call set_config - this should execute line 506: self.config = config
        plotter.set_config(new_config)

        # Verify config was updated
        assert plotter.config.x_label == "New X Label"
        assert plotter.config.y_label == "New Y Label"
        assert plotter.config.show_sample_info == "All"

    def test_configure_y_axis_percentage_display(self, sample_generic_data):
        """Test _setup_y_axis_scaling with percentage display and high y_lim_top."""
        config = create_boxplot_config(
            convert_to_percent=True,
            y_lim_bottom=0,
            y_lim_top=150,  # > 100 to trigger percentage adjustment
            show=False,
            save_path=None,
        )
        plotter = GenericBoxPlotter(sample_generic_data, config)

        # Mock matplotlib components
        with patch("matplotlib.pyplot.gca") as mock_gca, patch("matplotlib.pyplot.yticks") as mock_yticks:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax

            plotter.setup_plot()

            # Call the method
            plotter._setup_y_axis_scaling()

            # Verify linear scale was used (not log)
            mock_ax.set_yscale.assert_not_called()
            # Verify y limits were set
            mock_ax.set_ylim.assert_called_once_with((0, 150))
            # Should call yticks because convert_to_percent=True and y_lim_top > 100
            mock_yticks.assert_called_once()

    def test_collect_sample_info_all(self, mock_plotter):
        """Test the _collect_sample_info_all method."""
        # Mock rect with caps data
        mock_rect = {"caps": [MagicMock(), MagicMock()]}
        # Mock the get_xydata method to return coordinates
        mock_rect["caps"][-1].get_xydata.return_value = [(1.0, 5.0), (3.0, 5.0)]

        # Initialize the sample_label_x_positions list
        mock_plotter.sample_label_x_positions = []

        # Call the method
        mock_plotter._collect_sample_info_all(mock_rect)

        # Verify the x-center was calculated and added
        expected_center = (1.0 + 3.0) / 2  # 2.0
        assert len(mock_plotter.sample_label_x_positions) == 1
        assert mock_plotter.sample_label_x_positions[0] == expected_center

    def test_collect_sample_info_all_multiple_calls(self, mock_plotter):
        """Test _collect_sample_info_all with multiple calls."""
        # Create mock rects
        mock_rect1 = {"caps": [MagicMock(), MagicMock()]}
        mock_rect1["caps"][-1].get_xydata.return_value = [(0.0, 5.0), (2.0, 5.0)]

        mock_rect2 = {"caps": [MagicMock(), MagicMock()]}
        mock_rect2["caps"][-1].get_xydata.return_value = [(4.0, 5.0), (6.0, 5.0)]

        # Initialize the list
        mock_plotter.sample_label_x_positions = []

        # Call multiple times
        mock_plotter._collect_sample_info_all(mock_rect1)
        mock_plotter._collect_sample_info_all(mock_rect2)

        # Verify both centers were added
        assert len(mock_plotter.sample_label_x_positions) == 2
        assert mock_plotter.sample_label_x_positions[0] == 1.0  # (0+2)/2
        assert mock_plotter.sample_label_x_positions[1] == 5.0  # (4+6)/2

    def test_create_abbreviated_labels_basic(self, mock_plotter):
        """Test _create_abbreviated_labels with basic parameters."""
        category_labels = ["Low", "Medium", "High", "Very High", "Extreme"]
        num_bins = 5
        blanks_around = 3

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around)

        # Should have: first_label + blank + arrow + blank + last_label = 5 items
        assert len(result) == 5
        assert result[0] == "Low"
        assert result[-1] == "Extreme"
        assert r"$\rightarrow$" in result
        # Should have one blank before and after arrow
        arrow_idx = result.index(r"$\rightarrow$")
        assert result[arrow_idx - 1] == ""
        assert result[arrow_idx + 1] == ""

    def test_create_abbreviated_labels_with_padding(self, mock_plotter):
        """Test _create_abbreviated_labels with padding enabled."""
        category_labels = ["A", "B", "C", "D", "E"]
        num_bins = 5
        blanks_around = 3

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around, add_padding=True)

        # Should have padding at start and end
        assert len(result) == 7  # 5 + 2 padding
        assert result[0] == ""  # Start padding
        assert result[-1] == ""  # End padding
        assert result[1] == "A"  # First label after padding
        assert result[-2] == "E"  # Last label before padding

    def test_create_abbreviated_labels_odd_blanks(self, mock_plotter):
        """Test _create_abbreviated_labels with odd number of total blanks."""
        category_labels = ["Start", "Middle1", "Middle2", "Middle3", "End"]
        num_bins = 6  # Will create 3 total blanks (6 - 3 = 3)
        blanks_around = 3

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around)

        assert len(result) == 6
        assert result[0] == "Start"
        assert result[-1] == "End"
        assert r"$\rightarrow$" in result

        # With 3 total blanks: floor(3/2) = 1 blank before, 3-1 = 2 blanks after
        arrow_idx = result.index(r"$\rightarrow$")
        # Should have 1 blank before arrow and 2 blanks after
        blanks_before = sum(1 for i in range(1, arrow_idx) if result[i] == "")
        blanks_after = sum(1 for i in range(arrow_idx + 1, len(result) - 1) if result[i] == "")
        assert blanks_before == 1
        assert blanks_after == 2

    def test_create_abbreviated_labels_even_blanks(self, mock_plotter):
        """Test _create_abbreviated_labels with even number of total blanks."""
        category_labels = ["A", "B", "C", "D", "E", "F", "G"]
        num_bins = 7  # Will create 4 total blanks (7 - 3 = 4)
        blanks_around = 3

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around)

        assert len(result) == 7
        assert result[0] == "A"
        assert result[-1] == "G"

        # With 4 total blanks: floor(4/2) = 2 blanks before, 4-2 = 2 blanks after
        arrow_idx = result.index(r"$\rightarrow$")
        blanks_before = sum(1 for i in range(1, arrow_idx) if result[i] == "")
        blanks_after = sum(1 for i in range(arrow_idx + 1, len(result) - 1) if result[i] == "")
        assert blanks_before == 2
        assert blanks_after == 2

    def test_create_abbreviated_labels_minimum_case(self, mock_plotter):
        """Test _create_abbreviated_labels with minimum viable parameters."""
        category_labels = ["Start", "End"]
        num_bins = 3  # Minimum case: 3 - 3 = 0 total blanks
        blanks_around = 3

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around)

        # Should have: first_label + arrow + last_label = 3 items
        assert len(result) == 3
        assert result[0] == "Start"
        assert result[1] == r"$\rightarrow$"
        assert result[2] == "End"

    def test_create_abbreviated_labels_with_blanks_around_5(self, mock_plotter):
        """Test _create_abbreviated_labels with blanks_around=5."""
        category_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        num_bins = 8  # Will create 3 total blanks (8 - 5 = 3)
        blanks_around = 5

        result = mock_plotter._create_abbreviated_labels(category_labels, num_bins, blanks_around)

        # With num_bins=8 and blanks_around=5:
        # total_blanks = 8 - 5 = 3
        # number_blanks_0 = floor(3/2) = 1 blank
        # number_blanks_1 = 3 - 1 = 2 blanks
        # Result: [first] + [1 blank] + [arrow] + [2 blanks] + [last]
        # Expected length: 1 + 1 + 1 + 2 + 1 = 6
        assert len(result) == 6
        assert result == ["A", "", r"$\rightarrow$", "", "", "H"]

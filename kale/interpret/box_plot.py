# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Wenjie Zhao, mcsoft12138@outlook.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
# =============================================================================

"""
Boxplotter classes and related functions for Uncertainty Quantification Visualization.

Classes:
    SampleInfoMode: Enumeration for sample information display modes
    BoxPlotConfig: Comprehensive plotting configuration with display and styling options
    BoxPlotData: Data container for evaluation metrics grouped by bins
    BoxPlotter: Abstract base class for all plotters
    GenericBoxPlotter: Multi-model comparison boxplots
    PerModelBoxPlotter: Individual model performance analysis
    ComparingQBoxPlotter: Q-value comparison studies

Data Processing Architecture:
    BoxPlotDataProcessor: Abstract base class using template method pattern
    GenericBoxPlotDataProcessor: Specialized processor for multi-model comparisons
    PerModelBoxPlotDataProcessor: Specialized processor for per-model analysis
    ComparingQBoxPlotDataProcessor: Specialized processor for Q-value comparisons

Factory Functions:
    create_boxplot_config(): Create configuration objects with sensible defaults
    create_boxplot_data(): Create data containers with required parameters

The data processing system uses the template method pattern to provide a consistent
processing workflow while allowing customization through specialized subclasses.
Each processor implements specific logic for different visualization modes while
sharing common utility methods from the base class.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import colormaps

from kale.interpret.visualize import save_or_show_plot


class SampleInfoMode(Enum):
    """Enumeration for sample information display modes."""

    NONE = "None"
    ALL = "All"
    AVERAGE = "Average"


@dataclass
class BoxPlotConfig:
    """
    Comprehensive configuration class for boxplot parameters and styling.

    This dataclass provides fine-grained control over all aspects of uncertainty
    quantification boxplot visualization, including display settings, styling options,
    spacing configuration, and output parameters.

    Attributes are organized into logical groups:

    **Display & Behavior**: x_label, y_label, show_sample_info, save_path, show,
    convert_to_percent, to_log, show_individual_dots

    **Axis & Layout**: y_lim_top, y_lim_bottom, subplot_bottom, subplot_left

    **Typography**: font_size_label, font_size_tick, sample_info_fontsize, legend_fontsize

    **Visual Styling**: colormap, matplotlib_style, box_edge_color, median_color,
    mean_facecolor, hatch_type, dot_alpha_normal, dot_alpha_alt

    **Spacing & Dimensions**: All spacing parameters for fine-tuned layout control

    **Output Settings**: figure_size, save_dpi, show_dpi, bbox_inches, pad_inches

    Usage:
        Create instances using the create_boxplot_config() factory function for
        parameter validation and convenience, or instantiate directly for maximum control.

    Note:
        All parameters have sensible defaults optimized for uncertainty quantification
        visualization in medical imaging applications.
    """

    # Display settings
    x_label: str = "Uncertainty Thresholded Bin"
    y_label: str = "Error (%)"
    show_sample_info: str = "None"
    save_path: Optional[str] = None
    show: bool = False
    convert_to_percent: bool = True
    to_log: bool = False
    show_individual_dots: bool = True

    # Axis limits
    y_lim_top: int = 120
    y_lim_bottom: float = -0.1

    # Font settings
    font_size_label: int = 30
    font_size_tick: int = 30

    # Box settings
    width: float = 0.2
    detailed_mode: bool = False

    # Specific settings for comparing_q plots
    hatch_type: str = ""

    # Plot style
    matplotlib_style: str = "fivethirtyeight"

    # Colors and styling
    colormap: str = "Set1"
    box_edge_color: str = "black"
    box_linewidth: float = 1.0
    median_color: str = "crimson"
    median_linewidth: float = 3.0
    mean_facecolor: str = "crimson"
    mean_edgecolor: str = "black"
    mean_markersize: float = 10.0

    # Individual dots
    dot_alpha_normal: float = 0.75
    dot_alpha_alt: float = 0.2

    # Spacing values
    inner_spacing: float = 0.1
    middle_spacing: float = 0.02
    outer_spacing: float = 0.24
    gap_large: float = 0.25
    gap_small: float = 0.12
    outer_gap_small: float = 0.35
    outer_gap_large: float = 0.24
    comparing_q_inner_spacing: float = 0.02
    comparing_q_outer_spacing: float = 0.2

    # Spacing threshold configuration
    large_spacing_threshold: int = 9
    small_spacing_threshold: int = 10

    # Spacing values for different scenarios
    gap_large_for_large_spacing: float = 0.25
    gap_large_for_small_spacing: float = 0.35
    gap_small_for_large_spacing: float = 0.12
    gap_small_for_small_spacing: float = 0.25

    # Box widths
    default_box_width: float = 0.25
    comparing_q_width_base: float = 0.2

    # Font sizes
    sample_info_fontsize: int = 25
    legend_fontsize: int = 20

    # Layout
    legend_columnspacing: float = 2.0
    legend_bbox_anchor: Tuple[float, float] = (0.5, 1.18)
    subplot_bottom: float = 0.15
    subplot_left: float = 0.15

    # Figure saving/showing
    figure_size: Tuple[float, float] = (16.0, 10.0)
    save_dpi: int = 600
    show_dpi: int = 100
    bbox_inches: str = "tight"
    pad_inches: float = 0.1

    def set_params(self, **kwargs: Any) -> "BoxPlotConfig":
        """
        Set parameters on the BoxPlotConfig instance and return self for method chaining.

        This method allows for convenient parameter updating using a fluent interface.
        It validates that provided parameters are valid BoxPlotConfig attributes.

        Args:
            **kwargs (Any): Keyword arguments corresponding to BoxPlotConfig attributes.
                     Any valid BoxPlotConfig parameter can be set.

        Returns:
            BoxPlotConfig: Self reference for method chaining.

        Examples:
            Basic usage:
            >>> config = BoxPlotConfig()
            >>> config.set_params(x_label="Custom Label", y_label="Custom Y")

            Method chaining:
            >>> config = BoxPlotConfig().set_params(
            ...     x_label="Uncertainty Bins",
            ...     colormap="Set1",
            ...     font_size_label=25
            ... )

            Multiple updates:
            >>> config.set_params(show=True).set_params(save_path="output.png")

        Raises:
            AttributeError: If an invalid parameter name is provided.
        """
        # Get valid field names for validation
        valid_fields = {f.name for f in fields(self)}

        for param_name, param_value in kwargs.items():
            if param_name not in valid_fields:
                raise AttributeError(
                    f"'{param_name}' is not a valid BoxPlotConfig parameter. "
                    f"Valid parameters are: {sorted(valid_fields)}"
                )
            setattr(self, param_name, param_value)

        return self


@dataclass
class BoxPlotData:
    """
    Data container for boxplot inputs and metadata.

    This dataclass organizes evaluation metrics and associated metadata required
    for uncertainty quantification boxplot visualization. It handles multi-model,
    multi-uncertainty type data in a structured format optimized for efficient
    plotting and analysis.

    **Core Components:**

    **evaluation_data_by_bins**: The primary data structure containing evaluation metrics
    organized by uncertainty bins. Supports both single experiments and Q-value comparisons.

    **metadata fields**: uncertainty_categories, models, category_labels, num_bins provide
    context and labeling information for proper visualization.

    **Data Structure Patterns:**

    - **Single Experiment**: One dictionary mapping model_uncertainty keys to bin data
    - **Q-value Comparison**: List of dictionaries, each representing different Q thresholds
    - **Multi-model Analysis**: Keys like "ResNet50_epistemic", "VGG16_aleatoric"

    **Validation & Usage:**

    Use the create_boxplot_data() factory function for automatic validation and
    parameter inference. Direct instantiation is supported for advanced use cases
    where manual control is needed.

    **Design Philosophy:**

    Flexible data structure accommodates various uncertainty quantification workflows
    while maintaining consistency and type safety. Optimized for memory efficiency
    with large evaluation datasets.
    """

    evaluation_data_by_bins: List[Dict[str, List[List[float]]]]
    uncertainty_categories: Optional[List[List[str]]] = None
    models: Optional[List[str]] = None
    category_labels: Optional[List[str]] = None
    num_bins: Optional[int] = None

    def set_data(self, **kwargs: Any) -> "BoxPlotData":
        """
        Set data parameters on the BoxPlotData instance and return self for method chaining.

        This method allows for convenient data parameter updating using a fluent interface.
        It validates that provided parameters are valid BoxPlotData attributes.

        Args:
            **kwargs (Any): Keyword arguments corresponding to BoxPlotData attributes.
                     Valid parameters include:
                     - evaluation_data_by_bins: Primary evaluation data
                     - uncertainty_categories: Hierarchical uncertainty type organization
                     - models: Model identifiers for analysis
                     - category_labels: Human-readable x-axis labels
                     - num_bins: Total uncertainty bins

        Returns:
            BoxPlotData: Self reference for method chaining.

        Examples:
            Basic usage:
            >>> data = BoxPlotData(evaluation_data_by_bins=[{}])
            >>> data.set_data(models=["ResNet50", "VGG16"])

            Method chaining:
            >>> data = BoxPlotData(evaluation_data_by_bins=[{}]).set_data(
            ...     models=["ResNet50"],
            ...     uncertainty_categories=[["epistemic"]],
            ...     num_bins=5
            ... )

            Multiple updates:
            >>> data.set_data(models=["ResNet50"]).set_data(num_bins=5)

        Raises:
            AttributeError: If an invalid parameter name is provided.
        """
        # Get valid field names for validation
        valid_fields = {f.name for f in fields(self)}

        for param_name, param_value in kwargs.items():
            if param_name not in valid_fields:
                raise AttributeError(
                    f"'{param_name}' is not a valid BoxPlotData parameter. "
                    f"Valid parameters are: {sorted(valid_fields)}"
                )
            setattr(self, param_name, param_value)

        return self


def create_boxplot_config(**kwargs: Any) -> BoxPlotConfig:
    """
    Factory function to create BoxPlotConfig with parameter validation and convenient defaults.

    This function provides a convenient way to create BoxPlotConfig instances with
    validation and intelligent parameter handling. It accepts any valid BoxPlotConfig
    parameter as keyword arguments.

    Args:
        **kwargs (Any): Any BoxPlotConfig parameter to override defaults. Common parameters include:
            - x_label, y_label: Axis labels
            - colormap: Color scheme (e.g., "Set1", "tab10", "Pastel1", "Dark2")
            - font_size_label, font_size_tick: Typography settings
            - figure_size, save_dpi, show_dpi: Output configuration
            - show, save_path: Display and saving behavior

            For complete parameter list, see BoxPlotConfig class documentation.

    Returns:
        BoxPlotConfig: Configured instance ready for use with BoxPlotter classes.

    Examples:
        Basic usage:
        >>> config = create_boxplot_config(
        ...     x_label="Custom Bins",
        ...     y_label="Custom Error (%)",
        ...     colormap="Set1"
        ... )

        Publication-ready plots:
        >>> config = create_boxplot_config(
        ...     font_size_label=20,
        ...     figure_size=(12, 8),
        ...     save_dpi=300,
        ...     save_path="publication_plot.png"
        ... )

        Development/debugging:
        >>> config = create_boxplot_config(
        ...     show=True,
        ...     show_individual_dots=False,
        ...     y_lim_top=50
        ... )

    Raises:
        None: Any type or value errors will be raised by the BoxPlotConfig constructor.
    """
    # Only pass kwargs that match BoxPlotConfig fields
    boxplotconfig_field_names = {f.name for f in fields(BoxPlotConfig)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in boxplotconfig_field_names}
    return BoxPlotConfig(**filtered_kwargs)


def create_boxplot_data(
    evaluation_data_by_bins: List[Dict[str, List[List[float]]]],
    uncertainty_categories: Optional[List[List[str]]] = None,
    models: Optional[List[str]] = None,
    category_labels: Optional[List[str]] = None,
    num_bins: Optional[int] = None,
    **kwargs: Any,
) -> BoxPlotData:
    """
    Factory function to create validated BoxPlotData containers for uncertainty quantification visualization.

    This function creates BoxPlotData instances with automatic validation and intelligent
    parameter inference. It handles the complexity of organizing evaluation metrics for
    different plotting scenarios while ensuring data consistency.

    Args:
        evaluation_data_by_bins (List[Dict[str, List[List[float]]]]): Primary evaluation data organized by uncertainty bins.
            For single experiments: List with one dict mapping "model_uncertainty" to bin data.
            For Q-value comparisons: List of dicts, each representing different Q thresholds.
        uncertainty_categories (Optional[List[List[str]]], optional): Hierarchical uncertainty type organization.
            Examples: [["epistemic"], ["aleatoric"]] or [["S-MHA"], ["E-MHA"]]
            Default: Auto-inferred from data keys.
        models (Optional[List[str]], optional): Model identifiers for analysis and legend generation.
            Examples: ["ResNet50", "VGG16"] or ["baseline"]
            Default: Auto-inferred from data key prefixes.
        category_labels (Optional[List[str]], optional): Human-readable x-axis labels.
            Examples: ["B1", "B2", "B3"] or ["Q=5", "Q=10", "Q=15"]
            Default: Auto-generated based on data structure.
        num_bins (Optional[int], optional): Total uncertainty bins for validation and layout.
            Default: Auto-inferred from data structure.
        **kwargs (Any): Extended metadata for specialized analysis modes.

    Returns:
        BoxPlotData: Validated data container ready for visualization with any BoxPlotter class.

    Examples:
        Single model analysis:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=[{
        ...         "ResNet50_epistemic": [[0.1, 0.12], [0.15, 0.18]]
        ...     }],
        ...     models=["ResNet50"],
        ...     uncertainty_categories=[["epistemic"]]
        ... )

        Multi-model comparison:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=[{
        ...         "ResNet50_epistemic": [[0.1, 0.12], [0.15, 0.18]],
        ...         "VGG16_epistemic": [[0.11, 0.13], [0.16, 0.19]]
        ...     }],
        ...     models=["ResNet50", "VGG16"],
        ...     uncertainty_categories=[["epistemic"]]
        ... )

        Q-value threshold study:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=[
        ...         {"model_epistemic": [[0.1, 0.12]]},  # Q=5
        ...         {"model_epistemic": [[0.09, 0.11]]}, # Q=10
        ...         {"model_epistemic": [[0.08, 0.10]]}  # Q=15
        ...     ],
        ...     category_labels=["Q=5", "Q=10", "Q=15"]
        ... )

    Raises:
        None: Any type or value errors will be raised by the BoxPlotData constructor.
    """
    # Only pass kwargs that match BoxPlotData fields
    boxplotdata_field_names = {f.name for f in fields(BoxPlotData)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in boxplotdata_field_names}
    return BoxPlotData(
        evaluation_data_by_bins=evaluation_data_by_bins,
        uncertainty_categories=uncertainty_categories,
        models=models,
        category_labels=category_labels,
        num_bins=num_bins,
        **filtered_kwargs,
    )


class BoxPlotter(ABC):
    """
    Abstract base class for uncertainty quantification boxplot visualization.

    Attributes:
        data: BoxPlotData containing evaluation metrics organized by bins
        config: BoxPlotConfig with display settings and styling options
        ax: Current matplotlib axes object for plotting
        fig: Current matplotlib figure object
        legend_patches: List of legend elements (patches and lines)
        max_bin_height: Maximum y-value for proper axis scaling
        sample_label_x_positions: X-coordinates for sample count labels
        all_sample_percs: Sample percentage information for display

    Note:
        This is an abstract base class. Use specialized subclasses:
        - GenericBoxPlotter: For multi-model comparisons
        - PerModelBoxPlotter: For individual model analysis
        - ComparingQBoxPlotter: For Q-value threshold studies
    """

    def __init__(self, data: Optional[BoxPlotData] = None, config: Optional[BoxPlotConfig] = None):
        """
        Initialize BoxPlotter with optional data and configuration.

        Args:
            data (Optional[BoxPlotData], optional): BoxPlotData containing evaluation metrics.
                Can be set later via set_data(). Defaults to None.
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig for display settings.
                Uses defaults if not provided. Defaults to None.
        """
        self.data = data
        self.config = config or BoxPlotConfig()

        # Plotting state variables
        self.ax: Optional["matplotlib.axes.Axes"] = None
        self.fig: Optional["matplotlib.figure.Figure"] = None
        self.legend_patches: List[Union[patches.Patch, mlines.Line2D]] = []
        self.max_bin_height = 0.0
        self.sample_label_x_positions: List[float] = []
        self.all_sample_percs: List[Union[float, List[float]]] = []
        self.processed_data: Optional[List[Dict]] = None
        self.bin_label_locs: Optional[List[List[float]]] = None

    def set_data(self, data: BoxPlotData) -> None:
        """
        Update the data container for this plotter.

        Args:
            data (BoxPlotData): New BoxPlotData containing evaluation metrics to visualize.
        """
        self.data = data

    def set_config(self, config: BoxPlotConfig) -> None:
        """
        Update the configuration settings for this plotter.

        Args:
            config (BoxPlotConfig): New BoxPlotConfig with updated display and styling options.
        """
        self.config = config

    @abstractmethod
    def process_data(self) -> None:
        """
        Abstract method to process data for plotting.

        This method must be implemented by subclasses to handle their specific
        data processing requirements.
        """
        pass

    def setup_plot(self) -> None:
        """
        Initialize matplotlib plotting environment and reset internal state.

        This method:
        1. Applies the matplotlib style from config
        2. Gets current axes and disables x-axis grid
        3. Resets all internal state variables for a fresh plot

        Called automatically by draw_boxplot(), but can be called manually
        for custom plotting workflows.
        """
        plt.style.use(self.config.matplotlib_style)
        ax_temp = plt.gca()
        assert ax_temp is not None
        self.ax = ax_temp
        assert self.ax is not None
        self.ax.xaxis.grid(False)

        # Reset state
        self.legend_patches = []
        self.max_bin_height = 0.0
        self.sample_label_x_positions = []

    def draw_boxplot(self) -> None:
        """
        Default implementation for creating boxplot visualizations.

        This method provides a common implementation that works for most plotting modes
        (Generic and PerModel). Subclasses can override this method for specialized behavior
        or override specific hook methods to customize parts of the process.

        The default implementation:
        1. Sets up the plot
        2. Processes data if needed
        3. Creates color scheme and legend
        4. Draws boxplots with standard styling
        5. Handles sample information
        6. Formats and finalizes the plot
        """
        self.setup_plot()

        if not self.processed_data:
            self.process_data()

        assert self.processed_data is not None
        assert self.data is not None
        assert self.data.uncertainty_categories is not None

        colors = colormaps.get_cmap(self.config.colormap)(np.arange(len(self.data.uncertainty_categories) + 1))

        # Set up legend using hook method for customization
        self._setup_legend_for_mode(colors)

        # Draw boxplots
        for data_item in self.processed_data:
            rect = self._draw_single_boxplot_item(data_item, colors)

            # Handle sample information
            if self.config.show_sample_info == "All":
                self._collect_sample_info_all(rect)

        assert self.bin_label_locs is not None  # Should be set by process_data
        if self.config.show_sample_info == "Average":
            self._collect_sample_info_average()

        # Format and finalize plot using hook method for customization
        self._finalize_plot()

    def _setup_legend_for_mode(self, colors) -> None:
        """
        Hook method for setting up legend in different plotting modes.

        Args:
            colors: Color array from colormap

        Default implementation handles standard legend setup.
        Subclasses can override for specialized legend behavior.
        """
        if hasattr(self, "legend_info") and self.legend_info:
            for legend_item in self.legend_info:
                self._add_legend_patch(
                    colors[legend_item["color_idx"]],
                    legend_item["model_type"],
                    legend_item["uncertainty_type"],
                    legend_item["hatch_idx"],
                )

    def _draw_single_boxplot_item(self, data_item: Dict, colors) -> Dict[str, List[Any]]:
        """
        Hook method for drawing a single boxplot item.

        Args:
            data_item: Dictionary containing boxplot data and positioning
            colors: Color array from colormap

        Returns:
            Matplotlib boxplot dictionary

        Default implementation handles standard boxplot drawing.
        Subclasses can override for specialized drawing behavior.
        """
        assert self.data is not None
        if self.data.uncertainty_categories:
            # Safely get dot color, avoid index out of bounds
            dot_color_idx = len(colors) - 1
            dot_color = colors[dot_color_idx]

            return self._create_single_boxplot(
                data_item["data"],
                [data_item["x_position"]],
                data_item["width"],
                self.config.convert_to_percent,
                self.config.show_individual_dots,
                colors[data_item["color_idx"]],
                dot_color,
                data_item["hatch_idx"],
            )
        else:
            # Default color handling
            return self._create_single_boxplot(
                data_item["data"],
                [data_item["x_position"]],
                data_item["width"],
                self.config.convert_to_percent,
                self.config.show_individual_dots,
                "blue",  # Default color
                "red",  # Default dot color
                data_item.get("hatch_idx", 0),
            )

    def _finalize_plot(self) -> None:
        """
        Hook method for finalizing the plot.

        Default implementation uses standard parameters.
        Subclasses can override to customize finalization behavior.
        """
        assert self.bin_label_locs is not None
        assert self.data is not None
        assert self.data.category_labels is not None
        assert self.data.num_bins is not None
        assert self.data.uncertainty_categories is not None

        # Default show_all_ticks behavior - subclasses can override
        show_all_ticks = getattr(self.config, "detailed_mode", True)

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            self.data.num_bins,
            self.data.uncertainty_categories,
            comparing_q=False,
            show_all_ticks=show_all_ticks,
        )

    def _add_legend_patch(self, color: str, model_type: str, uncertainty_type: str, hatch_idx: int) -> None:
        """
        Create and register a legend patch for model-uncertainty combinations.

        Args:
            color (str): Color for the legend patch (hex code or named color).
            model_type (str): Model identifier for legend label.
            uncertainty_type (str): Uncertainty category for legend label.
            hatch_idx (int): Hatching pattern index (1 for hatched, 0 for solid).

        Note:
            Patches are stored in self.legend_patches for later legend creation.
        """
        if hatch_idx == 1:
            legend_patch = patches.Patch(
                facecolor=color,
                label=model_type + " " + uncertainty_type,
                hatch=self.config.hatch_type,
                edgecolor="black",
            )
        else:
            legend_patch = patches.Patch(facecolor=color, label=model_type + " " + uncertainty_type)
        self.legend_patches.append(legend_patch)

    def _create_single_boxplot(
        self,
        data: List[float],
        x_loc: List[float],
        width: float,
        convert_to_percent: bool,
        show_individual_dots: bool,
        box_color: str,
        dot_color: str,
        hatch_idx: int = 0,
        dot_alpha: float = 0.75,
    ) -> Dict[str, List[Any]]:
        """
        Create a single matplotlib boxplot with comprehensive styling and data visualization features.

        Args:
            data (List[float]): Numerical evaluation data for boxplot generation.
            x_loc (List[float]): X-axis positioning coordinates for boxplot placement.
            width (float): Box width parameter for visual appearance and spacing.
            convert_to_percent (bool): Data transformation flag for percentage display.
            show_individual_dots (bool): Individual data point overlay control.
            box_color (str): Box face color specification for visual distinction.
            dot_color (str): Individual data point color for scatter overlay.
            hatch_idx (int, optional): Hatching pattern application control flag. Defaults to 0.
            dot_alpha (float, optional): Transparency for individual dots. Defaults to 0.75.

        Returns:
            Dict[str, List[Any]]: Matplotlib boxplot dictionary containing plot elements and metadata.
                Structure: {"boxes": [], "medians": [], "means": [], "whiskers": [], "caps": []}
                Elements: Provides access to individual plot components for further customization
                Usage: Can be used to extract statistical information or apply additional styling
        """
        assert self.ax is not None

        # Convert data to percentage and filter out None values
        if convert_to_percent:
            displayed_data = [x * 100 for x in data if x is not None]
        else:
            displayed_data = [x for x in data if x is not None]

        rect = self.ax.boxplot(displayed_data, positions=x_loc, sym="", widths=width, showmeans=True, patch_artist=True)

        if show_individual_dots:
            # Add random "jitter" to x-axis
            x = np.random.normal(x_loc, 0.01, size=len(displayed_data))
            self.ax.plot(
                x,
                displayed_data,
                color=dot_color,
                marker=".",
                linestyle="None",
                alpha=dot_alpha,
            )

        # Set colors, patterns, median lines and mean markers
        for r in rect["boxes"]:
            r.set(color="black", linewidth=1)
            r.set(facecolor=box_color)

            if hatch_idx == 1:
                r.set_hatch(self.config.hatch_type)

        for median in rect["medians"]:
            median.set(color="crimson", linewidth=3)

        for mean in rect["means"]:
            mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=10)

        self.max_bin_height = max(max(rect["caps"][-1].get_ydata()), self.max_bin_height)

        return rect

    def _collect_sample_info_all(self, rect) -> None:
        """
        Collect and register sample distribution information for statistical annotation display(show_sample_info="All").

        Args:
            rect (Any): Matplotlib boxplot dictionary for geometric information extraction.

        Note:
            Method operates through side effects on instance variables and parameter lists.
        """
        (x_l, _), (x_r, _) = rect["caps"][-1].get_xydata()
        x_line_center = (x_l + x_r) / 2
        self.sample_label_x_positions.append(x_line_center)

    def _collect_sample_info_average(self) -> None:
        """
        Collect and register sample distribution information for statistical annotation display(show_sample_info="Average").
        """
        assert self.bin_label_locs is not None, "Bin label locations must be set before collecting sample info"
        for bin_label_loc in self.bin_label_locs:
            middle_x = float(np.mean(bin_label_loc))
            self.sample_label_x_positions.append(middle_x)

    def _display_sample_information(self) -> None:
        """
        Display sample percentage information above boxplots based on configuration settings.

        Handles both "Average" and "All" modes with appropriate positioning and formatting.
        Text is positioned above boxplots with alternating heights to prevent overlap.

        Note:
            Requires self.config.show_sample_info to be set to a valid mode.
            Uses self.all_sample_percs and self.sample_label_x_positions for placement.
        """
        assert self.ax is not None

        if self.config.show_sample_info == "None":
            return

        for idx_text, perc_info in enumerate(self.all_sample_percs):
            if self.config.show_sample_info == "Average":
                if isinstance(perc_info, list) and len(perc_info) >= 2:
                    self.ax.text(
                        self.sample_label_x_positions[idx_text],
                        self.max_bin_height * 0.8,  # Position
                        r"$\bf{PSB}$" + ": \n" + r"${} \pm$".format(perc_info[0]) + "\n" + r"${}$".format(perc_info[1]),
                        verticalalignment="bottom",  # Center with line bottom
                        horizontalalignment="center",  # Center with horizontal line
                        fontsize=self.config.sample_info_fontsize,
                    )
            elif self.config.show_sample_info == "All":
                label_height = self.max_bin_height + 2 * (idx_text % 2) + 1
                self.ax.text(
                    self.sample_label_x_positions[idx_text],
                    label_height,  # Position
                    r"$\bf{PSB}$" + ": \n" + str(perc_info) + "%",
                    verticalalignment="bottom",  # Center with line bottom
                    horizontalalignment="center",  # Center with horizontal line
                    fontsize=self.config.sample_info_fontsize,
                )

    def _setup_basic_axes_formatting(self, bin_label_locs: List[List[float]], show_all_ticks: bool) -> None:
        """
        Configure basic axis labels, ticks, and subplot adjustments.

        Args:
            bin_label_locs (List[float]): X-axis positions for tick mark placement.
            show_all_ticks (bool): Whether to show all tick marks.
        """
        assert self.ax is not None

        self.ax.set_xlabel(self.config.x_label, fontsize=self.config.font_size_label)
        self.ax.set_ylabel(self.config.y_label, fontsize=self.config.font_size_label)

        # Configure x-axis ticks
        if show_all_ticks:
            self.ax.set_xticks([item for sublist in bin_label_locs for item in sublist])
        else:
            self.ax.set_xticks([np.mean(locs) for locs in bin_label_locs])

        plt.subplots_adjust(bottom=self.config.subplot_bottom)
        plt.subplots_adjust(left=self.config.subplot_left)

        plt.xticks(fontsize=self.config.font_size_tick)
        plt.yticks(fontsize=self.config.font_size_tick)

    def _setup_x_axis_formatter(
        self, category_labels: List[str], num_bins: int, uncertainty_categories: List[List[str]], comparing_q: bool
    ) -> None:
        """
        Configure x-axis label formatting based on plot type and bin count.

        Args:
            category_labels (List[str]): Human-readable labels for x-axis categories.
            num_bins (int): Total number of uncertainty bins for layout optimization.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            comparing_q (bool): Q-value comparison mode flag.

        Note:
            Q-value comparison plots use simple direct labeling.
            Other modes use formatted labels with repetition multipliers.
        """
        assert self.ax is not None

        # Q-value comparison plots use simple direct labeling
        if comparing_q:
            self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(category_labels))
            return

        # Calculate label repetition multiplier for uncertainty categories
        multiplier = len(uncertainty_categories) * 2

        # Choose formatting strategy based on bin count to optimize readability
        formatted_labels = self._get_formatted_labels_for_bin_count(category_labels, num_bins)
        self.ax.xaxis.set_major_formatter(ticker.FixedFormatter(formatted_labels * multiplier))

    def _get_formatted_labels_for_bin_count(self, category_labels: List[str], num_bins: int) -> List[str]:
        """
        Select appropriate label formatting strategy based on the number of bins to ensure readability.

        Args:
            category_labels (List[str]): Original category labels.
            num_bins (int): Number of bins to determine formatting strategy.

        Returns:
            List[str]: Formatted labels optimized for the given bin count.
        """
        # Constants for better readability and maintainability
        SMALL_BIN_THRESHOLD = 5
        MEDIUM_BIN_THRESHOLD = 15

        if num_bins <= SMALL_BIN_THRESHOLD:
            # For few bins, show all labels except the last (which typically represents "remaining")
            return category_labels[:-1] if len(category_labels) > 1 else category_labels

        elif num_bins <= MEDIUM_BIN_THRESHOLD:
            # For medium bin counts, use abbreviated labels with moderate spacing
            return self._create_abbreviated_labels(
                category_labels=category_labels, num_bins=num_bins, blanks_around=3, add_padding=False
            )
        else:
            # For many bins, use abbreviated labels with extra padding to prevent overlap
            return self._create_abbreviated_labels(
                category_labels=category_labels, num_bins=num_bins, blanks_around=5, add_padding=True
            )

    def _create_abbreviated_labels(
        self, category_labels: List[str], num_bins: int, blanks_around: int, add_padding: bool = False
    ) -> List[str]:
        """
        Create abbreviated label lists for x-axis when there are too many bins.

        Args:
            category_labels (List[str]): Original category labels.
            num_bins (int): Total number of bins.
            blanks_around (int): Number of blanks to distribute around arrow (3 or 5).
            add_padding (bool, optional): Whether to add empty padding at start and end. Defaults to False.

        Returns:
            List[str]: Abbreviated label list with blanks and arrow.
        """
        total_blanks = num_bins - blanks_around
        number_blanks_0 = ["" for _ in range(math.floor(total_blanks / 2))]
        number_blanks_1 = ["" for _ in range(total_blanks - len(number_blanks_0))]

        labels = [category_labels[0]] + number_blanks_0 + [r"$\rightarrow$"] + number_blanks_1 + [category_labels[-1]]

        if add_padding:
            labels = [""] + labels + [""]

        return labels

    def _setup_y_axis_scaling(self) -> None:
        """
        Configure y-axis scaling, limits, and tick formatting based on configuration.

        Handles both linear and logarithmic scaling options, and adjusts tick spacing
        for percentage display when needed.
        """
        assert self.ax is not None

        if self.config.to_log:
            self.ax.set_yscale("symlog", base=2)
            self.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            self.ax.set_ylim(self.config.y_lim_bottom, self.config.y_lim_top)
        else:
            self.ax.set_ylim((self.config.y_lim_bottom, self.config.y_lim_top))

        # Adjust y-ticks for percentage display
        if self.config.convert_to_percent and self.config.y_lim_top > 100:
            plt.yticks(np.arange(0, self.config.y_lim_top, 20))

    def _setup_legend(self) -> None:
        """
        Configure and display the plot legend with mean/median symbols and sample info.

        Creates legend entries for:
        - Model-uncertainty combinations from self.legend_patches
        - Mean symbol (red triangle)
        - Median symbol (red line)
        - Sample info notation (if showing averages)
        """
        assert self.ax is not None

        # Add mean symbol to legend
        red_triangle_mean = mlines.Line2D(
            [],
            [],
            color=self.config.mean_facecolor,
            marker="^",
            markeredgecolor=self.config.mean_edgecolor,
            linestyle="None",
            markersize=self.config.mean_markersize,
            label="Mean",
        )
        self.legend_patches.append(red_triangle_mean)

        # Add median symbol to legend
        red_line_median = mlines.Line2D(
            [],
            [],
            color=self.config.median_color,
            marker="",
            markeredgecolor="black",
            markersize=10,
            label="Median",
        )
        self.legend_patches.append(red_line_median)

        # Add sample info legend entry if showing average
        if self.config.show_sample_info == "Average":
            self.legend_patches.append(patches.Patch(color="none", label=r"$\bf{PSB}$" + r": % Samples per Bin"))

        # Create and display legend
        num_cols_legend = math.ceil(len(self.legend_patches) / 2)
        self.ax.legend(
            handles=self.legend_patches,
            fontsize=self.config.legend_fontsize,
            ncol=num_cols_legend,
            columnspacing=self.config.legend_columnspacing,
            loc="upper center",
            bbox_to_anchor=self.config.legend_bbox_anchor,
            fancybox=True,
            shadow=False,
        )

    def _format_and_finalize_plot(
        self,
        bin_label_locs: List[List[float]],
        category_labels: List[str],
        num_bins: int,
        uncertainty_categories: List[List[str]],
        comparing_q: bool = False,
        show_all_ticks: bool = True,
    ) -> None:
        """
        Apply comprehensive formatting, styling, and finalization to complete boxplot visualization.

        This method orchestrates the complete plot finalization process by calling specialized
        sub-functions for each formatting aspect, making the code more maintainable and testable.

        Args:
            bin_label_locs (List[List[float]]): X-axis positions for tick mark and label placement.
            category_labels (List[str]): Human-readable labels for x-axis categories.
            num_bins (int): Total number of uncertainty bins for layout optimization.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            comparing_q (bool, optional): Q-value comparison mode flag. Defaults to False.
            show_all_ticks (bool, optional): Whether to show all tick marks. Defaults to True.
        """
        assert self.ax is not None

        # Execute formatting steps in logical order
        self._display_sample_information()
        self._setup_basic_axes_formatting(bin_label_locs, show_all_ticks=show_all_ticks)
        self._setup_x_axis_formatter(category_labels, num_bins, uncertainty_categories, comparing_q)
        self._setup_y_axis_scaling()
        self._setup_legend()
        save_or_show_plot(
            save_path=self.config.save_path,
            show=self.config.show,
            fig_size=self.config.figure_size,
            save_dpi=self.config.save_dpi,
            show_dpi=self.config.show_dpi,
            bbox_inches=self.config.bbox_inches,
            pad_inches=self.config.pad_inches,
        )


class GenericBoxPlotter(BoxPlotter):
    """
    Multi-model comparison boxplot visualization for uncertainty analysis.

    This specialized plotter creates side-by-side boxplot comparisons across different
    models and uncertainty types. Ideal for comparative studies where you need to
    evaluate multiple models under the same uncertainty quantification framework.

    The plotter organizes data by uncertainty categories first, then by models within
    each category, creating clear visual comparisons while maintaining consistent
    color coding and spacing.

    Data Organization:
    - Groups boxplots by uncertainty type (e.g., epistemic vs aleatoric)
    - Within each group, shows different models side-by-side
    - Maintains consistent spacing and coloring across comparisons

    Use Cases:
    - Comparing model performance across uncertainty types
    - Evaluating the impact of different architectures on uncertainty estimation
    - Publishing comparative studies with multiple model baselines

    Example:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=multi_model_results,
        ...     uncertainty_categories=[['epistemic'], ['aleatoric']],
        ...     models=['ResNet50', 'VGG16', 'DenseNet']
        ... )
        >>> config = create_boxplot_config(colormap='Set1')
        >>> plotter = GenericBoxPlotter(data, config)
        >>> plotter.draw_boxplot()  # Creates comparison plot
    """

    def __init__(self, data: Optional[BoxPlotData] = None, config: Optional[BoxPlotConfig] = None):
        """
        Initialize GenericBoxPlotter for multi-model comparisons.

        Args:
            data (Optional[BoxPlotData], optional): BoxPlotData with evaluation_data_by_bins,
                uncertainty_categories, and models. Defaults to None.
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig for styling and
                display options. Defaults to None.
        """
        super().__init__(data, config)
        self.legend_info: Optional[List[Dict]] = None

    def process_data(self) -> None:
        """
        Process data for generic plotting mode.

        Validates required data fields and processes evaluation data for multi-model
        comparison visualization using GenericBoxPlotDataProcessor.

        Raises:
            ValueError: If BoxPlotData is not set or missing required fields for generic mode.
        """
        if not self.data:
            raise ValueError("BoxPlotData must be set before processing")

        assert self.data is not None

        # Additional checks for required fields in generic mode
        if not all(
            [self.data.evaluation_data_by_bins, self.data.uncertainty_categories, self.data.models, self.data.num_bins]
        ):
            raise ValueError(
                "GenericBoxPlotter requires evaluation_data_by_bins, uncertainty_categories, models, and num_bins"
            )

        assert self.data.evaluation_data_by_bins is not None
        assert self.data.uncertainty_categories is not None
        assert self.data.models is not None
        assert self.data.num_bins is not None

        processor = GenericBoxPlotDataProcessor(self.config)
        self.processed_data, self.legend_info, self.bin_label_locs, self.all_sample_percs = processor.process_data(
            self.data.evaluation_data_by_bins[0],
            self.data.uncertainty_categories,
            self.data.models,
            self.data.num_bins,
        )

    def _finalize_plot(self) -> None:
        """
        Override finalization to use detailed_mode for show_all_ticks in generic plots.
        """
        assert self.bin_label_locs is not None
        assert self.data is not None
        assert self.data.category_labels is not None
        assert self.data.num_bins is not None
        assert self.data.uncertainty_categories is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            self.data.num_bins,
            self.data.uncertainty_categories,
            comparing_q=False,
            show_all_ticks=self.config.detailed_mode,
        )


class PerModelBoxPlotter(BoxPlotter):
    """
    Individual model performance analysis with grouped uncertainty visualization.

    This specialized plotter focuses on detailed analysis of individual models by
    organizing boxplots by model first, then by uncertainty type within each model.
    Perfect for in-depth model analysis and understanding uncertainty patterns
    within specific architectures.

    Data Organization:
    - Primary grouping by model (e.g., ResNet50, VGG16)
    - Secondary grouping by uncertainty type within each model
    - Bins displayed consecutively for each model-uncertainty combination

    Use Cases:
    - Deep dive analysis of individual model performance
    - Understanding uncertainty patterns within specific architectures
    - Model-specific calibration and validation studies
    - Detailed performance reports for single-model deployments

    Example:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=resnet_results,
        ...     uncertainty_categories=[['epistemic'], ['aleatoric']],
        ...     models=['ResNet50']  # Focus on single model
        ... )
        >>> config = create_boxplot_config(colormap='Set1')
        >>> plotter = PerModelBoxPlotter(data, config)
        >>> plotter.draw_boxplot()  # Creates detailed per-model analysis
    """

    def __init__(self, data: Optional[BoxPlotData] = None, config: Optional[BoxPlotConfig] = None):
        """
        Initialize PerModelBoxPlotter for detailed individual model analysis.

        Args:
            data (Optional[BoxPlotData], optional): BoxPlotData configured for per-model analysis.
                Defaults to None.
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig with appropriate styling
                for detailed views. Defaults to None.
        """
        super().__init__(data, config)
        self.legend_info: Optional[List[Dict]] = None

    def process_data(self) -> None:
        """
        Process data for per-model plotting mode.

        Validates required data fields and processes evaluation data for individual
        model performance analysis using PerModelBoxPlotDataProcessor.

        Raises:
            ValueError: If BoxPlotData is not set or missing required fields for per-model mode."""
        if not self.data:
            raise ValueError("BoxPlotData must be set before processing")

        assert self.data is not None

        # Additional checks for required fields in per-model mode
        if not all(
            [self.data.evaluation_data_by_bins, self.data.uncertainty_categories, self.data.models, self.data.num_bins]
        ):
            raise ValueError(
                "PerModelBoxPlotter requires evaluation_data_by_bins, uncertainty_categories, models, and num_bins"
            )

        assert self.data.evaluation_data_by_bins is not None
        assert self.data.uncertainty_categories is not None
        assert self.data.models is not None
        assert self.data.num_bins is not None

        processor = PerModelBoxPlotDataProcessor(self.config)
        self.processed_data, self.legend_info, self.bin_label_locs, self.all_sample_percs = processor.process_data(
            self.data.evaluation_data_by_bins[0],
            self.data.uncertainty_categories,
            self.data.models,
            self.data.num_bins,
        )

    def _finalize_plot(self) -> None:
        """
        Override finalization to always use show_all_ticks=True for per-model plots.
        """
        assert self.bin_label_locs is not None
        assert self.data is not None
        assert self.data.category_labels is not None
        assert self.data.num_bins is not None
        assert self.data.uncertainty_categories is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            self.data.num_bins,
            self.data.uncertainty_categories,
            comparing_q=False,
            show_all_ticks=True,
        )


class ComparingQBoxPlotter(BoxPlotter):
    """
    Quantile threshold comparison visualization for binning strategy analysis.

    This specialized plotter compares the impact of different quantile thresholds
    (Q values) on uncertainty quantification performance. Essential for studies
    that need to optimize binning strategies or understand threshold sensitivity.

    Data Organization:
    - Each Q value (e.g., Q=5, Q=10, Q=15) gets its own dataset
    - Boxplots show performance distribution for each threshold
    - Special hatched styling distinguishes Q-comparison plots
    - Progressive box width reduction for visual clarity

    Key Features:
    - Handles list of evaluation datasets (one per Q value)
    - Applies distinctive hatching patterns for Q-studies
    - Optimized x-axis labeling for threshold values
    - Consistent single-model, single-uncertainty focus

    Use Cases:
    - Optimizing uncertainty quantile thresholds
    - Sensitivity analysis for binning strategies
    - Publication plots for threshold selection studies
    - Hyperparameter tuning for uncertainty quantification

    Example:
        >>> # Compare Q=5 vs Q=10 vs Q=15 thresholds
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=[q5_results, q10_results, q15_results],
        ...     uncertainty_categories=[['epistemic']],
        ...     models=['ResNet50'],
        ...     category_labels=['Q=5', 'Q=10', 'Q=15']
        ... )
        >>> config = create_boxplot_config(hatch_type='///', colormap='Set1')
        >>> plotter = ComparingQBoxPlotter(data, config)
        >>> plotter.draw_boxplot()
    """

    def __init__(self, data: Optional[BoxPlotData] = None, config: Optional[BoxPlotConfig] = None):
        """
        Initialize ComparingQBoxPlotter for quantile threshold comparison.

        Args:
            data (Optional[BoxPlotData], optional): BoxPlotData with evaluation_data_by_bins
                for Q-value comparison. Defaults to None.
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig with hatch_type and
                color for Q-plot styling. Defaults to None.
        """
        super().__init__(data, config)
        self.uncertainty_type: Optional[str] = None

    def process_data(self) -> None:
        """
        Process data for comparing Q plotting mode.

        Validates required data fields and processes evaluation data for Q-value
        threshold comparison using ComparingQBoxPlotDataProcessor.

        Raises:
            ValueError: If BoxPlotData is not set or missing required fields for Q-comparison mode."""
        if not self.data:
            raise ValueError("BoxPlotData must be set before processing")

        assert self.data is not None

        if not all([self.data.evaluation_data_by_bins, self.data.uncertainty_categories, self.data.models]):
            raise ValueError(
                "For comparing_q plots, data must include evaluation_data_by_bins, uncertainty_categories, and models"
            )

        assert self.data.evaluation_data_by_bins is not None
        assert self.data.uncertainty_categories is not None
        assert self.data.models is not None
        assert self.data.category_labels is not None

        processor = ComparingQBoxPlotDataProcessor(self.config)
        self.processed_data, _, self.bin_label_locs, self.all_sample_percs = processor.process_data(
            self.data.evaluation_data_by_bins,
            self.data.uncertainty_categories,
            self.data.models,
            self.data.category_labels,
        )

        # Extract parameters for plotting
        assert self.data is not None
        assert self.data.uncertainty_categories is not None
        assert self.data.models is not None
        self.uncertainty_type = self.data.uncertainty_categories[0][0]
        self.model_type = self.data.models[0]

    def _setup_legend_for_mode(self, colors) -> None:
        """
        Override legend setup for Q-comparison mode with specialized styling.

        Args:
            colors: Color array from colormap (used for color selection)
        """
        assert self.data is not None
        assert self.data.uncertainty_categories is not None
        colors = colormaps.get_cmap(self.config.colormap)(np.arange(3))
        color = (
            colors[0]
            if self.data.uncertainty_categories[0][0] == "S-MHA"
            else colors[1]
            if self.data.uncertainty_categories[0][0] == "E-MHA"
            else colors[2]
        )

        # Set legend for comparing_q mode with special hatched styling
        if self.uncertainty_type and self.model_type and self.config.hatch_type:
            legend_patch = patches.Patch(
                hatch=self.config.hatch_type,
                facecolor=color,
                label=self.model_type + " " + self.uncertainty_type,
                edgecolor="black",
            )
            self.legend_patches.append(legend_patch)

    def _draw_single_boxplot_item(self, data_item: Dict, colors) -> Dict[str, List[Any]]:
        """
        Override boxplot drawing for Q-comparison mode with specialized styling.

        Args:
            data_item: Dictionary containing boxplot data and positioning
            colors: Color array from colormap

        Returns:
            Matplotlib boxplot dictionary
        """
        assert self.data is not None
        assert self.data.uncertainty_categories is not None
        colors = colormaps.get_cmap(self.config.colormap)(np.arange(3))
        color = (
            colors[0]
            if self.data.uncertainty_categories[0][0] == "S-MHA"
            else colors[1]
            if self.data.uncertainty_categories[0][0] == "E-MHA"
            else colors[2]
        )

        # Draw boxplots with special hatched style and fixed parameters for Q-comparison
        return self._create_single_boxplot(
            data_item["data"],
            [data_item["x_position"]],
            data_item["width"],
            self.config.convert_to_percent,
            self.config.show_individual_dots,
            color,
            "crimson",  # Fixed dot color for Q-comparison
            hatch_idx=1,  # Always use hatching for Q-comparison
            dot_alpha=0.2,  # Reduced alpha for Q-comparison
        )

    def _finalize_plot(self) -> None:
        """
        Override finalization for Q-comparison plots with specialized parameters.
        """
        assert self.uncertainty_type is not None  # Should be set by process_data
        assert self.bin_label_locs is not None
        assert self.data is not None
        assert self.data.category_labels is not None

        # Create specialized parameters for Q-comparison plots
        uncertainty_types_for_plot = [[self.uncertainty_type]]
        num_bins_for_plot = self.data.num_bins
        assert num_bins_for_plot is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            num_bins_for_plot,
            uncertainty_types_for_plot,
            comparing_q=True,  # Specialized flag for Q-comparison
            show_all_ticks=False,  # Always False for Q-comparison
        )


class BoxPlotDataProcessor(ABC):
    """
    Abstract base class for boxplot data processing using template method pattern.

    This class defines the common data processing workflow for different boxplot modes,
    while allowing subclasses to customize specific operations through method overrides.

    The template method pattern ensures consistent processing flow while enabling
    flexible customization for different visualization requirements.
    """

    def __init__(self, config: Optional[BoxPlotConfig] = None):
        """
        Initialize the processor with default state.

        Args:
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig for display settings.
                Uses defaults if not provided. Defaults to None.

        Sets up instance variables for input data, processing state, and results.
        All variables are initialized to None or empty collections.
        """
        # Input data
        self.evaluation_data_by_bin: Optional[Dict[str, List[List[float]]]] = None
        self.evaluation_data_by_bins: Optional[List[Dict[str, List[List[float]]]]] = None
        self.uncertainty_categories: Optional[List[List[str]]] = None
        self.models: Optional[List[str]] = None
        self.num_bins: Optional[int] = None
        self.category_labels: Optional[List[str]] = None
        self.config: BoxPlotConfig = config or BoxPlotConfig()

        # Processing state
        self.outer_min_x_loc: float = 0.0
        self.middle_min_x_loc: float = 0.0
        self.inner_min_x_loc: float = 0.0

        # Results
        self.processed_data: List[Dict] = []
        self.legend_info: Optional[List[Dict]] = []
        self.bin_label_locs: List[List[float]] = []
        self.all_sample_percs: List[Union[float, List[float]]] = []
        self.all_box_x_positions: List[float] = []

    def process_data(
        self, *args, **kwargs
    ) -> Tuple[List[Dict], Union[List[Dict], None], List[List[float]], List[Union[float, List[float]]]]:
        """
        Template method that defines the main data processing algorithm.

        This method coordinates the overall processing flow by calling specific
        hook methods that subclasses can override to customize behavior.

        Args:
            *args: Variable positional arguments specific to each processing mode.
            **kwargs: Variable keyword arguments specific to each processing mode.

        Returns:
            Tuple[List[Dict], Union[List[Dict], None], List[List[float]], List[float]]: A tuple containing:
                - processed_data: List of dictionaries with boxplot data and positioning
                - legend_info: Legend information (None for Q-comparison mode)
                - bin_label_locs: X-axis positions for bin labels
                - all_sample_percs: Sample percentage information

        Raises:
            ValueError: If processing results in empty data or invalid state.
        """
        # Initialize processing parameters
        self._initialize_processing_parameters(*args, **kwargs)

        # Execute main processing loop (varies by subclass)
        self._execute_processing_loop()

        # Finalize and validate results
        self._finalize_processing_results()

        return self.processed_data, self.legend_info, self.bin_label_locs, self.all_sample_percs

    @staticmethod
    def create_data_item(
        data: List[float],
        x_position: float,
        width: float,
        color_idx: int,
        model_type: str,
        uncertainty_type: str,
        hatch_idx: int,
        bin_idx: int,
        model_data: List[List[float]],
        percent_size: float,
        **extra_fields,
    ) -> Dict:
        """
        Create standard data item dictionary for boxplot processing.

        Args:
            data (List[float]): Boxplot data values.
            x_position (float): X-axis position for the boxplot.
            width (float): Box width for the boxplot.
            color_idx (int): Color index for styling.
            model_type (str): Model identifier.
            uncertainty_type (str): Uncertainty category.
            hatch_idx (int): Hatching pattern index.
            bin_idx (int): Bin index.
            model_data (List[List[float]]): Original model data.
            **extra_fields: Additional fields to include in the data item.

        Returns:
            Dict: Standardized data item dictionary with all required fields.

        Examples:
            >>> item = create_data_item([0.1, 0.2], 1.0, 0.2, 0, "ResNet50", "epistemic", 0, 0, model_data)
        """
        item = {
            "data": data,
            "x_position": x_position,
            "width": width,
            "color_idx": color_idx,
            "model_type": model_type,
            "uncertainty_type": uncertainty_type,
            "hatch_idx": hatch_idx,
            "bin_idx": bin_idx,
            "model_data": model_data,
            "percent_size": percent_size,
        }
        item.update(extra_fields)  # Add additional fields
        return item

    # Abstract hook methods that subclasses must implement
    @abstractmethod
    def _initialize_processing_parameters(self, *args, **kwargs) -> None:
        """
        Initialize the processing parameters from input arguments.

        Subclasses should override this method to extract and store relevant
        parameters as instance variables.

        Args:
            *args: Variable positional arguments specific to processing mode.
            **kwargs: Variable keyword arguments specific to processing mode.
        """
        pass

    @abstractmethod
    def _execute_processing_loop(self) -> None:
        """
        Execute the main data processing loop.

        Each subclass implements its specific processing logic directly.
        Results should be stored in self.processed_data, self.legend_info,
        and self.bin_label_locs.
        """
        pass

    def _finalize_processing_results(self) -> None:
        """
        Optional hook for result validation and finalization.

        Default implementation performs basic validation.
        Subclasses can override for additional processing.

        Raises:
            ValueError: If processing resulted in empty data or bin label locations.
        """
        if not self.processed_data:
            raise ValueError("Processing resulted in empty data")
        if not self.bin_label_locs:
            raise ValueError("Processing resulted in empty bin label locations")

    @staticmethod
    def _find_data_key(
        evaluation_data_by_bin: Dict[str, List[List[float]]], model_type: str, uncertainty_type: str
    ) -> str:
        """
        Locate the appropriate data key for model-uncertainty combination.

        Args:
            evaluation_data_by_bin (Dict[str, List[List[float]]]): Dictionary with keys like
                "ResNet50_epistemic", "VGG16_aleatoric".
            model_type (str): Model identifier to search for (e.g., "ResNet50", "VGG16").
            uncertainty_type (str): Uncertainty category to search for (e.g., "epistemic", "aleatoric").

        Returns:
            str: First matching dictionary key containing both model and uncertainty type.

        Raises:
            KeyError: If no key contains both model_type and uncertainty_type.

        Examples:
            >>> key = _find_data_key(data, "ResNet50", "epistemic")  # Returns "ResNet50_epistemic"
            >>> key = _find_data_key(data, "VGG16", "aleatoric")     # Returns "VGG16_aleatoric"
        """
        matching_keys = [
            key for key in evaluation_data_by_bin.keys() if (model_type in key) and (uncertainty_type in key)
        ]
        if not matching_keys:
            raise KeyError(f"No matching key found: model_type='{model_type}', uncertainty_type='{uncertainty_type}'")
        return matching_keys[0]

    def _calculate_spacing_adjustment(self, is_large_spacing: bool) -> float:
        """
        Calculate spacing adjustment value based on bin count using configurable parameters.

        Args:
            is_large_spacing (bool): Whether to use large spacing configuration.

        Returns:
            float: Spacing adjustment value based on configuration and bin count.

        Note:
            Uses config thresholds and spacing values to determine appropriate spacing.
        """
        assert self.config is not None, "Configuration must be set before calculating spacing adjustment"

        # Select threshold based on spacing type
        threshold = self.config.large_spacing_threshold if is_large_spacing else self.config.small_spacing_threshold

        # Select spacing values based on spacing type
        large_val = (
            self.config.gap_large_for_large_spacing if is_large_spacing else self.config.gap_large_for_small_spacing
        )
        small_val = (
            self.config.gap_small_for_large_spacing if is_large_spacing else self.config.gap_small_for_small_spacing
        )

        assert self.num_bins is not None, "Number of bins must be set before calculating spacing adjustment"
        return large_val if self.num_bins > threshold else small_val

    def _extract_bin_data(self, model_data: List[List[float]], bin_idx: int) -> List[float]:
        """
        Extract data from specific bin from model data.

        Args:
            model_data (List[List[float]]): Model data organized by bins.
            bin_idx (int): Index of the bin to extract data from.

        Returns:
            List[float]: Extracted bin data, optionally filtered for None values.

        Examples:
            >>> data = [[0.1, 0.2], [0.3, None, 0.4]]
            >>> _extract_bin_data(data, 0)  # Returns [0.1, 0.2]
            >>> _extract_bin_data(data, 1)  # Returns [0.3, 0.4]
        """
        assert self.config is not None, "Configuration must be set before extracting bin data"

        if self.config.detailed_mode:
            return [x for x in model_data[bin_idx] if x is not None]
        else:
            return model_data[bin_idx]

    def _calculate_sample_percs(self, model_data: List[List[float]], bin_data: List[float]) -> float:
        """
        Calculate the percentage of samples in a specific bin compared to the total samples.

        Args:
            model_data (List[List[float]]): The model data organized by bins.
            bin_data (List[float]): The data for the specific bin.

        Returns:
            float: The percentage of samples in the bin.
        """
        if self.config.show_sample_info != "None":
            flattened_model_data = [x for xss in model_data for x in xss]
            percent_size = np.round(len(bin_data) / len(flattened_model_data) * 100, 1)
            if self.config.show_sample_info == "All":
                self.all_sample_percs.append(percent_size)
            return percent_size
        return 0

    def _calculate_average_sample_info(self, average_samples_per_bin: List[float]) -> None:
        """
        Calculate and store average sample percentages for the current configuration.

        Args:
            average_samples_per_bin (List[float]): List of sample percentages for each bin.
        """
        mean_perc = np.round(np.mean(average_samples_per_bin), 1)
        std_perc = np.round(np.std(average_samples_per_bin), 1)
        self.all_sample_percs.append([mean_perc, std_perc])

    # Shared utility methods for common processing patterns
    def _process_single_item(
        self,
        uncertainty_type: str,
        model_type: str,
        bin_idx: int,
        uncertainty_idx: int,
        hatch_idx: int,
        width: float,
        evaluation_data_override: Optional[Dict] = None,
    ) -> Dict:
        """
        Process a single data item and return the result.

        Args:
            uncertainty_type (str): Uncertainty category identifier.
            model_type (str): Model identifier.
            bin_idx (int): Bin index to process.
            uncertainty_idx (int): Uncertainty type index for coloring.
            hatch_idx (int): Hatching pattern index.
            width (float): Box width for the item.
            evaluation_data_override (Dict, optional): Override data for Q-comparison mode.
                Defaults to None.

        Returns:
            Dict: Processed data item dictionary with all required fields.
        """
        # Use override data for Q-comparison mode, otherwise use instance data
        evaluation_data = evaluation_data_override or self.evaluation_data_by_bin
        assert evaluation_data is not None, "Evaluation data must be available"

        # Extract data using base class method
        dict_key = BoxPlotDataProcessor._find_data_key(evaluation_data, model_type, uncertainty_type)
        model_data = evaluation_data[dict_key]

        bin_data = self._extract_bin_data(model_data, bin_idx)

        # Calculate position from instance variables
        x_loc = self.outer_min_x_loc + self.middle_min_x_loc + self.inner_min_x_loc

        percent_size = self._calculate_sample_percs(model_data, bin_data)

        # Create and return data item
        return self.create_data_item(
            bin_data,
            x_loc,
            width,
            uncertainty_idx,
            model_type,
            uncertainty_type,
            hatch_idx,
            bin_idx,
            model_data,
            percent_size,
        )

    def _process_and_store_single_item(
        self,
        uncertainty_type: str,
        model_type: str,
        bin_idx: int,
        uncertainty_idx: int,
        hatch_idx: int,
        width: float,
        evaluation_data_override: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        """
        Process single item and store in processed_data, return x position.

        Args:
            uncertainty_type (str): Uncertainty category identifier.
            model_type (str): Model identifier.
            bin_idx (int): Bin index to process.
            uncertainty_idx (int): Uncertainty type index for coloring.
            hatch_idx (int): Hatching pattern index.
            width (float): Box width for the item.
            evaluation_data_override (Dict, optional): Override data for Q-comparison mode.
                Defaults to None.

        Returns:
            tuple[float, float]: X-axis position and percent size of the processed item.
        """
        data_item = self._process_single_item(
            uncertainty_type, model_type, bin_idx, uncertainty_idx, hatch_idx, width, evaluation_data_override
        )
        self.processed_data.append(data_item)
        return data_item["x_position"], data_item["percent_size"]

    def _collect_legend_info_for_first_bin(
        self,
        bin_idx: int,
        uncertainty_idx: int,
        model_type: str,
        uncertainty_type: str,
        hatch_idx: int,
    ) -> None:
        """
        Collect legend information if this is the first bin.

        Args:
            bin_idx (int): Current bin index.
            uncertainty_idx (int): Uncertainty type index for coloring.
            model_type (str): Model identifier for legend.
            uncertainty_type (str): Uncertainty category for legend.
            hatch_idx (int): Hatching pattern index."""
        if bin_idx == 0 and self.legend_info is not None:
            legend_item = {
                "color_idx": uncertainty_idx,
                "model_type": model_type,
                "uncertainty_type": uncertainty_type,
                "hatch_idx": hatch_idx,
            }
            self.legend_info.append(legend_item)

    def _process_and_collect_positions(
        self,
        uncertainty_type: str,
        uncertainty_idx: int,
        items_data: List[Tuple[int, str, int]],  # (bin_idx, model_type, hatch_idx)
        width: float,
    ) -> List[float]:
        """
        Process items and collect their positions with common logic.

        Args:
            uncertainty_type (str): Uncertainty category identifier.
            uncertainty_idx (int): Uncertainty type index for coloring.
            items_data (List[Tuple[int, str, int]]): List of (bin_idx, model_type, hatch_idx) tuples.
            width (float): Box width for all items.
        """
        box_x_positions = []
        average_samples_per_bin = []

        assert self.config is not None, "Configuration must be set before processing and collecting positions"
        for bin_idx, model_type, hatch_idx in items_data:
            x_position, percent_size = self._process_and_store_single_item(
                uncertainty_type, model_type, bin_idx, uncertainty_idx, hatch_idx, width
            )
            box_x_positions.append(x_position)
            average_samples_per_bin.append(percent_size)

            # Collect legend info for first bin
            self._collect_legend_info_for_first_bin(bin_idx, uncertainty_idx, model_type, uncertainty_type, hatch_idx)

            # Update inner position with spacing
            self.inner_min_x_loc += self.config.inner_spacing + width

        if self.config.show_sample_info == "Average":
            self._calculate_average_sample_info(average_samples_per_bin)

        self.bin_label_locs.append(box_x_positions)

        return box_x_positions


class GenericBoxPlotDataProcessor(BoxPlotDataProcessor):
    """
    Data processor for generic multi-model boxplot visualization.

    Implements uncertainty -> bins -> models processing order for comparing
    multiple models across different uncertainty types.
    """

    def _initialize_processing_parameters(
        self,
        evaluation_data_by_bin: Dict[str, List[List[float]]],
        uncertainty_categories: List[List[str]],
        models: List[str],
        num_bins: int,
    ) -> None:
        """
        Initialize processing parameters for generic mode.

        Args:
            evaluation_data_by_bin (Dict[str, List[List[float]]]): Evaluation data organized by bins.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            models (List[str]): List of model identifiers.
            num_bins (int): Total number of uncertainty bins.
        """
        self.evaluation_data_by_bin = evaluation_data_by_bin
        self.uncertainty_categories = uncertainty_categories
        self.models = models
        self.num_bins = num_bins
        self.legend_info = []  # Generic mode has legend info

    def _execute_processing_loop(self) -> None:
        """
        Execute generic processing: uncertainty -> bins -> models.

        Processes data in the order: for each uncertainty type, for each bin,
        for each model. This creates grouped comparisons by uncertainty type.
        """
        assert self.config is not None, "Configuration must be set before processing"
        assert self.uncertainty_categories is not None, "Uncertainty categories must be set"
        assert self.models is not None, "Models must be set"

        detailed_mode = self.config.detailed_mode

        for uncertainty_idx, uncert_pair in enumerate(self.uncertainty_categories):
            uncertainty_type = uncert_pair[0]

            # Process each bin for this uncertainty type
            assert self.num_bins is not None, "Number of bins must be set before processing"
            for bin_idx in range(self.num_bins):
                width = self.config.default_box_width

                # Prepare items data for processing
                items_data = [(bin_idx, model_type, hatch_idx) for hatch_idx, model_type in enumerate(self.models)]

                # Process all models for this bin using shared method
                self._process_and_collect_positions(uncertainty_type, uncertainty_idx, items_data, width)

                self.middle_min_x_loc += self.config.middle_spacing

            # Apply spacing between uncertainty types
            spacing_adjustment = self._calculate_spacing_adjustment(True)
            if detailed_mode:
                self.middle_min_x_loc += spacing_adjustment
            else:
                self.outer_min_x_loc += spacing_adjustment


class PerModelBoxPlotDataProcessor(BoxPlotDataProcessor):
    """
    Data processor for per-model boxplot visualization.

    Implements uncertainty -> models -> bins processing order for detailed
    individual model analysis with model-first grouping.
    """

    def _initialize_processing_parameters(
        self,
        evaluation_data_by_bin: Dict[str, List[List[float]]],
        uncertainty_categories: List[List[str]],
        models: List[str],
        num_bins: int,
    ) -> None:
        """
        Initialize processing parameters for per-model mode.

        Args:
            evaluation_data_by_bin (Dict[str, List[List[float]]]): Evaluation data organized by bins.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            models (List[str]): List of model identifiers.
            num_bins (int): Total number of uncertainty bins.
        """
        self.evaluation_data_by_bin = evaluation_data_by_bin
        self.uncertainty_categories = uncertainty_categories
        self.models = models
        self.num_bins = num_bins
        self.legend_info = []  # Per-model mode has legend info

    def _execute_processing_loop(self) -> None:
        """
        Execute per-model processing: uncertainty -> models -> bins.

        Processes data in the order: for each uncertainty type, for each model,
        for each bin. This creates detailed per-model analysis with model-first grouping.
        """
        assert self.config is not None, "Configuration must be set before processing"
        assert self.uncertainty_categories is not None, "Uncertainty categories must be set"
        assert self.models is not None, "Models must be set"

        for uncertainty_idx, uncert_pair in enumerate(self.uncertainty_categories):
            uncertainty_type = uncert_pair[0]

            # Process each model for this uncertainty type
            for hatch_idx, model_type in enumerate(self.models):
                width = self.config.default_box_width

                # Prepare items data for processing
                assert self.num_bins is not None, "Number of bins must be set before processing"
                items_data = [(bin_idx, model_type, hatch_idx) for bin_idx in range(self.num_bins)]

                # Process all bins for this model using shared method
                self._process_and_collect_positions(uncertainty_type, uncertainty_idx, items_data, width)

                # Apply spacing between models
                spacing_adjustment = self._calculate_spacing_adjustment(False)
                self.middle_min_x_loc += spacing_adjustment

            # Apply final spacing after each uncertainty type
            self.outer_min_x_loc += self.config.outer_spacing


class ComparingQBoxPlotDataProcessor(BoxPlotDataProcessor):
    """
    Data processor for Q-value comparison boxplot visualization.

    Implements specialized processing for quantile threshold comparison
    studies with progressive width reduction.
    """

    def __init__(self, config: Optional[BoxPlotConfig] = None):
        """
        Initialize ComparingQ processor with specific parameters.

        Args:
            config (Optional[BoxPlotConfig], optional): BoxPlotConfig for display settings.
                Uses defaults if not provided. Defaults to None.

        Extends base processor initialization with Q-comparison specific variables
        including evaluation_data_by_bins list and category_labels.
        """
        super().__init__(config)
        self.evaluation_data_by_bins = []
        self.category_labels = []
        self.uncertainty_type = ""
        self.model_type = ""

    def _initialize_processing_parameters(
        self,
        evaluation_data_by_bins: List[Dict[str, List[List[float]]]],
        uncertainty_categories: List[List[str]],
        model: List[str],
        category_labels: List[str],
    ) -> None:
        """
        Initialize processing parameters for comparing Q mode.

        Args:
            evaluation_data_by_bins (List[Dict[str, List[List[float]]]]): List of evaluation
                data, one per Q-value.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            model (List[str]): Single model identifier (list with one element).
            category_labels (List[str]): Labels for Q-value categories.
        """
        self.evaluation_data_by_bins = evaluation_data_by_bins
        self.uncertainty_categories = uncertainty_categories
        self.models = model
        self.category_labels = category_labels
        self.uncertainty_type = uncertainty_categories[0][0]
        self.model_type = model[0]
        self.legend_info = None  # Q mode has no legend info

    @staticmethod
    def _calculate_progressive_width(base_width: float, q_idx: int) -> float:
        """
        Calculate progressive width reduction for Q-value comparison.

        Args:
            base_width (float): Base width for the first Q-value.
            q_idx (int): Index of the current Q-value.

        Returns:
            float: Reduced width for visual distinction between Q-values.

        Note:
            Uses (4/5)^q_idx formula for progressive width reduction.
        """
        return base_width * (4 / 5) ** q_idx

    def _process_single_q_value(
        self, q_idx: int, evaluation_data_by_bin: Dict[str, List[List[float]]], base_width: float
    ) -> List[float]:
        """
        Process all bins for a single Q value.

        Args:
            q_idx (int): Index of the current Q-value.
            evaluation_data_by_bin (Dict[str, List[List[float]]]): Evaluation data for this Q-value.
            base_width (float): Base width for width calculation.

        Returns:
            List[float]: X-axis positions of all boxes for this Q-value.
        """
        box_x_positions = []
        average_samples_per_bin = []

        # Get data for this Q value
        dict_key = BoxPlotDataProcessor._find_data_key(evaluation_data_by_bin, self.model_type, self.uncertainty_type)
        model_data = evaluation_data_by_bin[dict_key]

        assert self.config is not None, "Configuration must be set before processing and collecting positions"
        # Process each bin for this Q value
        for bin_idx in range(len(model_data)):
            # Progressive width reduction
            width = ComparingQBoxPlotDataProcessor._calculate_progressive_width(base_width, q_idx)

            x_position, percent_size = self._process_and_store_single_item(
                self.uncertainty_type,
                self.model_type,
                bin_idx,
                0,
                0,
                width,
                evaluation_data_override=evaluation_data_by_bin,
            )
            box_x_positions.append(x_position)
            average_samples_per_bin.append(percent_size)

            self.inner_min_x_loc += self.config.inner_spacing + width

        if self.config.show_sample_info == "Average":
            self._calculate_average_sample_info(average_samples_per_bin)

        return box_x_positions

    def _apply_q_spacing_and_store_labels(self, box_x_positions: List[float]) -> None:
        """
        Apply outer spacing and store bin label locations for Q comparison.

        Args:
            box_x_positions (List[float]): X-axis positions of boxes for current Q-value.
        """
        assert self.config is not None, "Configuration must be set before applying spacing"

        self.outer_min_x_loc += self.config.comparing_q_outer_spacing
        self.bin_label_locs.append(box_x_positions)

    def _execute_processing_loop(self) -> None:
        """
        Execute Q-comparison processing loop with progressive width.

        Processes each Q-value with progressive width reduction for visual distinction.
        Uses specialized Q-value processing methods.
        """
        assert self.config is not None, "Configuration must be set before processing"
        assert self.category_labels is not None, "Category labels must be set"
        assert self.evaluation_data_by_bins is not None, "Evaluation data must be set"

        base_width = self.config.comparing_q_width_base

        for q_idx, _ in enumerate(self.category_labels):
            evaluation_data_by_bin = self.evaluation_data_by_bins[q_idx]

            # Process this Q value
            box_x_positions = self._process_single_q_value(q_idx, evaluation_data_by_bin, base_width)

            # Apply spacing and store labels for this Q value
            self._apply_q_spacing_and_store_labels(box_x_positions)

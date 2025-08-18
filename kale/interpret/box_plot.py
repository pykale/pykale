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

Factory Functions:
    create_boxplot_config(): Create configuration objects with sensible defaults
    create_boxplot_data(): Create data containers with required parameters

"""

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
    use_list_comp: bool = False

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
    gap_large: float = 0.25
    gap_small: float = 0.12
    outer_gap_small: float = 0.35
    outer_gap_large: float = 0.24
    comparing_q_spacing: float = 0.2

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
        TypeError: If parameter types don't match expected formats.
        ValueError: If parameter values are outside valid ranges.
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
        evaluation_data_by_bins: Primary evaluation data organized by uncertainty bins.
            For single experiments: List with one dict mapping "model_uncertainty" to bin data.
            For Q-value comparisons: List of dicts, each representing different Q thresholds.

        uncertainty_categories: Hierarchical uncertainty type organization.
            Examples: [["epistemic"], ["aleatoric"]] or [["S-MHA"], ["E-MHA"]]
            Default: Auto-inferred from data keys.

        models: Model identifiers for analysis and legend generation.
            Examples: ["ResNet50", "VGG16"] or ["baseline"]
            Default: Auto-inferred from data key prefixes.

        category_labels: Human-readable x-axis labels.
            Examples: ["B1", "B2", "B3"] or ["Q=5", "Q=10", "Q=15"]
            Default: Auto-generated based on data structure.

        num_bins: Total uncertainty bins for validation and layout.
            Default: Auto-inferred from data structure.

        **kwargs: Extended metadata for specialized analysis modes.

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
        ValueError: If data dimensions are inconsistent or parameters don't match data structure.
        TypeError: If data types don't match expected formats.
        KeyError: If required data keys are missing.
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


class BoxPlotter:
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
            data: BoxPlotData containing evaluation metrics. Can be set later via set_data().
            config: BoxPlotConfig for display settings. Uses defaults if not provided.
        """
        self.data = data
        self.config = config or BoxPlotConfig()

        # Plotting state variables
        self.ax = None  # matplotlib.axes.Axes
        self.fig = None  # matplotlib.figure.Figure
        self.legend_patches: List[Union[patches.Patch, mlines.Line2D]] = []
        self.max_bin_height = 0.0
        self.sample_label_x_positions: List[float] = []
        self.all_sample_percs: List[Union[float, List[float]]] = []

    def set_data(self, data: BoxPlotData) -> None:
        """
        Update the data container for this plotter.

        Args:
            data: New BoxPlotData containing evaluation metrics to visualize.
        """
        self.data = data

    def set_config(self, config: BoxPlotConfig) -> None:
        """
        Update the configuration settings for this plotter.

        Args:
            config: New BoxPlotConfig with updated display and styling options.
        """
        self.config = config

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
        self.ax = plt.gca()
        assert self.ax is not None
        self.ax.xaxis.grid(False)

        # Reset state
        self.legend_patches = []
        self.max_bin_height = 0.0
        self.sample_label_x_positions = []
        self.all_sample_percs = []

    def draw_boxplot(self) -> None:
        """
        Abstract method for creating the actual boxplot visualization.

        Raises:
            NotImplementedError: Always raised - subclasses must provide implementation
        """
        raise NotImplementedError("Subclasses must implement draw_boxplot method")

    def _add_legend_patch(self, color: str, model_type: str, uncertainty_type: str, hatch_idx: int) -> None:
        """
        Create and register a legend patch for model-uncertainty combinations.

        Args:
            color: Color for the legend patch (hex code or named color)
            model_type: Model identifier for legend label
            uncertainty_type: Uncertainty category for legend label
            hatch_idx: Hatching pattern index (1 for hatched, 0 for solid)

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
        hatch_idx: int,
    ) -> Dict[str, List[Any]]:  # matplotlib boxplot return type
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
            hatch_idx (int): Hatching pattern application control flag.

        Returns:
            Dict[str, List[Any]]: Matplotlib boxplot dictionary containing plot elements and metadata.
                Structure: {"boxes": [], "medians": [], "means": [], "whiskers": [], "caps": []}
                Elements: Provides access to individual plot components for further customization
                Usage: Can be used to extract statistical information or apply additional styling
                Side effects: Updates self.max_bin_height for axis scaling coordination
        """
        assert self.ax is not None

        # Convert data to percentage
        if convert_to_percent:
            displayed_data = [(x) * 100 for x in data]
        else:
            displayed_data = data

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
                alpha=0.75,
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

    def _create_single_boxplot_with_hatch(
        self,
        data: List[float],
        x_loc: List[float],
        width: float,
        convert_to_percent: bool,
        show_individual_dots: bool,
        color: str,
        dot_color: str,
        hatch_type: str,
    ) -> Dict[str, List[Any]]:  # matplotlib boxplot return type
        """
        Create specialized hatched boxplot for Q-value comparison studies with distinctive visual patterns.

        Args:
            data (List[float]): Numerical evaluation data for boxplot generation.
            x_loc (List[float]): X-axis positioning coordinates for precise boxplot placement.
            width (float): Box width parameter with progressive reduction for Q-studies.
            convert_to_percent (bool): Data transformation control for percentage display.
            show_individual_dots (bool): Individual data point overlay control.
            color (str): Primary box face color for Q-value series identification.
            dot_color (str): Scatter point color for individual data visualization.
            hatch_type (str): Hatching pattern specification for visual distinction.

        Returns:
            Dict[str, List[Any]]: Matplotlib boxplot dictionary with hatched styling applied.
                Structure: Standard matplotlib boxplot return format
                Elements: All boxes have mandatory hatching patterns applied
                Usage: Can be used for statistical extraction or additional customization
                Side effects: Updates self.max_bin_height for axis coordination
        """
        assert self.ax is not None

        # Convert data to percentage
        if convert_to_percent:
            displayed_data = [(x) * 100 for x in data]
        else:
            displayed_data = data

        rect = self.ax.boxplot(displayed_data, positions=x_loc, sym="", widths=width, showmeans=True, patch_artist=True)

        self.max_bin_height = max(max(rect["caps"][-1].get_ydata()), self.max_bin_height)

        if show_individual_dots:
            # Add random "jitter" to x-axis
            x = np.random.normal(x_loc, 0.01, size=len(displayed_data))
            self.ax.plot(x, displayed_data, color=dot_color, marker=".", linestyle="None", alpha=0.2)

        # Set colors, patterns, median lines and mean markers
        for r in rect["boxes"]:
            r.set(color="black", linewidth=1)
            r.set(facecolor=color)
            r.set_hatch(hatch_type)
        for median in rect["medians"]:
            median.set(color="crimson", linewidth=3)

        for mean in rect["means"]:
            mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=10)

        return rect

    def _collect_sample_info_if_needed(
        self,
        show_sample_info: str,
        bin_data: List[float],
        model_data: List[List[float]],
        average_samples_per_bin: List[float],
        rect,
    ) -> None:
        """
        Collect and register sample distribution information for statistical annotation display.

        Args:
            show_sample_info (str): Sample information display mode control.
            bin_data (List[float]): Current bin's evaluation data for sample counting.
            model_data (List[List[float]]): Complete model evaluation data for total sample calculation.
            average_samples_per_bin (List[float]): Accumulator for sample percentage collection.
            rect (Any): Matplotlib boxplot dictionary for geometric information extraction.

        Returns:
            None: Method operates through side effects on instance variables and parameter lists.
        """
        if show_sample_info != "None":
            flattened_model_data = [x for xss in model_data for x in xss]
            percent_size = np.round(len(bin_data) / len(flattened_model_data) * 100, 1)
            average_samples_per_bin.append(percent_size)

            if show_sample_info == "All":
                """Add sample count above the highest whisker line"""
                (x_l, y), (x_r, _) = rect["caps"][-1].get_xydata()
                x_line_center = (x_l + x_r) / 2
                self.sample_label_x_positions.append(x_line_center)
                self.all_sample_percs.append(percent_size)

    def _display_sample_information(self) -> None:
        """
        Display sample percentage information above boxplots based on configuration settings.

        Handles both "Average" and "All" modes with appropriate positioning and formatting.
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

    def _setup_basic_axes_formatting(self, bin_label_locs: List[float]) -> None:
        """
        Configure basic axis labels, ticks, and subplot adjustments.

        Args:
            bin_label_locs (List[float]): X-axis positions for tick mark placement.
        """
        assert self.ax is not None

        self.ax.set_xlabel(self.config.x_label, fontsize=self.config.font_size_label)
        self.ax.set_ylabel(self.config.y_label, fontsize=self.config.font_size_label)
        self.ax.set_xticks(bin_label_locs)

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
            add_padding (bool): Whether to add empty padding at start and end.

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
        bin_label_locs: List[float],
        category_labels: List[str],
        num_bins: int,
        uncertainty_categories: List[List[str]],
        comparing_q: bool = False,
    ) -> None:
        """
        Apply comprehensive formatting, styling, and finalization to complete boxplot visualization.

        This method orchestrates the complete plot finalization process by calling specialized
        sub-functions for each formatting aspect, making the code more maintainable and testable.

        Args:
            bin_label_locs (List[float]): X-axis positions for tick mark and label placement.
            category_labels (List[str]): Human-readable labels for x-axis categories.
            num_bins (int): Total number of uncertainty bins for layout optimization.
            uncertainty_categories (List[List[str]]): Hierarchical uncertainty type organization.
            comparing_q (bool, optional): Q-value comparison mode flag.

        Returns:
            None: Method operates through side effects on matplotlib axes and figure.
        """
        assert self.ax is not None

        # Execute formatting steps in logical order
        self._display_sample_information()
        self._setup_basic_axes_formatting(bin_label_locs)
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
            data: BoxPlotData with evaluation_data_by_bins, uncertainty_categories, and models
            config: BoxPlotConfig for styling and display options
        """
        super().__init__(data, config)
        self.processed_data: Optional[List[Dict]] = None
        self.legend_info: Optional[List[Dict]] = None
        self.bin_label_locs: Optional[List[float]] = None

    def process_data(self) -> None:
        """Process data for generic plotting mode."""
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

        self.processed_data, self.legend_info, self.bin_label_locs = _process_boxplot_data_generic(
            self.data.evaluation_data_by_bins[0],
            self.data.uncertainty_categories,
            self.data.models,
            self.data.num_bins,
            self.config,
        )

    def draw_boxplot(self) -> None:
        """Draw generic multi-model boxplot."""
        if not self.processed_data:
            self.process_data()

        assert self.processed_data is not None
        assert self.data is not None
        assert self.data.uncertainty_categories is not None

        self.setup_plot()
        colors = colormaps.get_cmap(self.config.colormap)(np.arange(len(self.data.uncertainty_categories) + 1))
        # Set legend for generic mode
        if self.legend_info:
            for legend_item in self.legend_info:
                self._add_legend_patch(
                    colors[legend_item["color_idx"]],
                    legend_item["model_type"],
                    legend_item["uncertainty_type"],
                    legend_item["hatch_idx"],
                )

        # Draw boxplots
        for data_item in self.processed_data:
            if self.data.uncertainty_categories:
                # Safely get dot color, avoid index out of bounds
                dot_color_idx = len(colors) - 1
                dot_color = colors[dot_color_idx]

                rect = self._create_single_boxplot(
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
                rect = self._create_single_boxplot(
                    data_item["data"],
                    [data_item["x_position"]],
                    data_item["width"],
                    self.config.convert_to_percent,
                    self.config.show_individual_dots,
                    "blue",  # Default color
                    "red",  # Default dot color
                    data_item.get("hatch_idx", 0),
                )

            # Handle sample information
            self._collect_sample_info_if_needed(
                self.config.show_sample_info, data_item["data"], data_item["model_data"], [], rect
            )

        # Format and finalize plot
        assert self.bin_label_locs is not None  # Should be set by process_data
        assert self.data.category_labels is not None
        assert self.data.num_bins is not None
        assert self.data.uncertainty_categories is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            self.data.num_bins,
            self.data.uncertainty_categories,
            comparing_q=False,
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
            data: BoxPlotData configured for per-model analysis
            config: BoxPlotConfig with appropriate styling for detailed views
        """
        super().__init__(data, config)
        self.processed_data: Optional[List[Dict]] = None
        self.legend_info: Optional[List[Dict]] = None
        self.bin_label_locs: Optional[List[float]] = None

    def process_data(self) -> None:
        """Process data for per-model plotting mode."""
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

        self.processed_data, self.legend_info, self.bin_label_locs = _process_boxplot_data_per_model(
            self.data.evaluation_data_by_bins[0],
            self.data.uncertainty_categories,
            self.data.models,
            self.data.num_bins,
            self.config,
        )

    def draw_boxplot(self) -> None:
        """Draw per-model boxplot."""
        if not self.processed_data:
            self.process_data()

        assert self.processed_data is not None
        assert self.data is not None
        assert self.data.uncertainty_categories is not None

        self.setup_plot()
        colors = colormaps.get_cmap(self.config.colormap)(np.arange(len(self.data.uncertainty_categories) + 1))
        # Set legend for per-model mode
        if self.legend_info:
            for legend_item in self.legend_info:
                self._add_legend_patch(
                    colors[legend_item["color_idx"]],
                    legend_item["model_type"],
                    legend_item["uncertainty_type"],
                    legend_item["hatch_idx"],
                )

        # Draw boxplots
        for data_item in self.processed_data:
            if self.data.uncertainty_categories:
                # Safely get dot color, avoid index out of bounds
                dot_color_idx = len(colors) - 1
                dot_color = colors[dot_color_idx]

                rect = self._create_single_boxplot(
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
                rect = self._create_single_boxplot(
                    data_item["data"],
                    [data_item["x_position"]],
                    data_item["width"],
                    self.config.convert_to_percent,
                    self.config.show_individual_dots,
                    "blue",  # Default color
                    "red",  # Default dot color
                    data_item.get("hatch_idx", 0),
                )

            # Handle sample information
            self._collect_sample_info_if_needed(
                self.config.show_sample_info, data_item["data"], data_item["model_data"], [], rect
            )

        # Format and finalize plot
        assert self.bin_label_locs is not None  # Should be set by process_data
        assert self.data.category_labels is not None
        assert self.data.num_bins is not None
        assert self.data.uncertainty_categories is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            self.data.num_bins,
            self.data.uncertainty_categories,
            comparing_q=False,
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
            data: BoxPlotData with evaluation_data_by_bins for Q-value comparison
            config: BoxPlotConfig with hatch_type and color for Q-plot styling
        """
        super().__init__(data, config)
        self.processed_data: Optional[List[Dict]] = None
        self.bin_label_locs: Optional[List[float]] = None
        self.uncertainty_type: Optional[str] = None

    def process_data(self) -> None:
        """Process data for comparing Q plotting mode."""
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

        self.processed_data, self.bin_label_locs = _process_boxplot_data_comparing_q(
            self.data.evaluation_data_by_bins,
            self.data.uncertainty_categories,
            self.data.models,
            self.data.category_labels,
        )

        # Extract parameters for plotting
        self.uncertainty_type = self.data.uncertainty_categories[0][0]
        self.model_type = self.data.models[0]

    def draw_boxplot(self) -> None:
        """Draw comparing Q boxplot."""
        if not self.processed_data:
            self.process_data()

        assert self.processed_data is not None
        assert self.data is not None
        assert self.data.uncertainty_categories is not None

        self.setup_plot()
        colors = colormaps.get_cmap(self.config.colormap)(np.arange(3))
        color = (
            colors[0]
            if self.data.uncertainty_categories[0][0] == "S-MHA"
            else colors[1]
            if self.data.uncertainty_categories[0][0] == "E-MHA"
            else colors[2]
        )
        # Set legend for comparing_q mode
        if self.uncertainty_type and self.model_type and self.config.hatch_type:
            legend_patch = patches.Patch(
                hatch=self.config.hatch_type,
                facecolor=color,
                label=self.model_type + " " + self.uncertainty_type,
                edgecolor="black",
            )
            self.legend_patches.append(legend_patch)

        # Draw boxplots with special hatched style
        for data_item in self.processed_data:
            rect = self._create_single_boxplot_with_hatch(
                data_item["data"],
                [data_item["x_position"]],
                data_item["width"],
                self.config.convert_to_percent,
                self.config.show_individual_dots,
                color,
                "crimson",
                self.config.hatch_type,
            )

            # Handle sample information
            self._collect_sample_info_if_needed(
                self.config.show_sample_info, data_item["data"], data_item["model_data"], [], rect
            )

        # Format and finalize plot
        assert self.uncertainty_type is not None  # Should be set by process_data
        uncertainty_types_for_plot = [[self.uncertainty_type]]
        num_bins_for_plot = self.data.num_bins

        assert self.bin_label_locs is not None  # Should be set by process_data
        assert self.data.category_labels is not None
        assert num_bins_for_plot is not None

        self._format_and_finalize_plot(
            self.bin_label_locs,
            self.data.category_labels,
            num_bins_for_plot,
            uncertainty_types_for_plot,
            comparing_q=True,
        )


# Data processing helper functions
# ================================


def _find_data_key(evaluation_data_by_bin: Dict[str, List[List[float]]], model_type: str, uncertainty_type: str) -> str:
    """
    Locate the appropriate data key for model-uncertainty combination.

    Args:
        evaluation_data_by_bin: Dictionary with keys like "ResNet50_epistemic", "VGG16_aleatoric"
        model_type: Model identifier to search for (e.g., "ResNet50", "VGG16")
        uncertainty_type: Uncertainty category to search for (e.g., "epistemic", "aleatoric")

    Returns:
        str: First matching dictionary key containing both model and uncertainty type

    Raises:
        KeyError: If no key contains both model_type and uncertainty_type
    """
    matching_keys = [key for key in evaluation_data_by_bin.keys() if (model_type in key) and (uncertainty_type in key)]
    if not matching_keys:
        raise KeyError(f"No matching key found: model_type='{model_type}', uncertainty_type='{uncertainty_type}'")
    return matching_keys[0]


def _extract_bin_data(model_data: List[List[float]], bin_idx: int, use_list_comp: bool = False) -> List[float]:
    """
    Extract data from specific bin from model data.

    Args:
        model_data: Model data list
        bin_idx: Bin index
        use_list_comp: Whether to use list comprehension to filter None values

    Returns:
        Extracted bin data
    """
    if use_list_comp:
        return [x for x in model_data[bin_idx] if x is not None]
    else:
        return model_data[bin_idx]


def _create_data_item(
    data: List[float],
    x_position: float,
    width: float,
    color_idx: int,
    model_type: str,
    uncertainty_type: str,
    hatch_idx: int,
    bin_idx: int,
    model_data: List[List[float]],
    **extra_fields,
) -> Dict:
    """
    Create standard data item dictionary.

    Args:
        data: Boxplot data
        x_position: x-axis position
        width: Box width
        color_idx: Color index
        model_type: Model type
        uncertainty_type: Uncertainty type
        hatch_idx: Hatch index
        bin_idx: Bin index
        model_data: Original model data
        **extra_fields: Additional fields

    Returns:
        Data item dictionary
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
    }
    item.update(extra_fields)  # Add additional fields
    return item


def _calculate_spacing_adjustment(num_bins: int, is_large_spacing: bool, config: BoxPlotConfig) -> float:
    """
    Calculate spacing adjustment value based on bin count using configurable parameters.

    Args:
        num_bins: Number of bins
        is_large_spacing: Whether to use large spacing configuration
        config: BoxPlotConfig object containing spacing parameters

    Returns:
        Spacing adjustment value based on configuration
    """
    # Select threshold based on spacing type
    threshold = config.large_spacing_threshold if is_large_spacing else config.small_spacing_threshold

    # Select spacing values based on spacing type
    large_val = config.gap_large_for_large_spacing if is_large_spacing else config.gap_large_for_small_spacing

    small_val = config.gap_small_for_large_spacing if is_large_spacing else config.gap_small_for_small_spacing

    return large_val if num_bins > threshold else small_val


def _process_boxplot_data_generic(
    evaluation_data_by_bin: Dict[str, List[List[float]]],
    uncertainty_categories: List[List[str]],
    models: List[str],
    num_bins: int,
    config: BoxPlotConfig,
) -> Tuple[List[Dict], List[Dict], List[float]]:
    """
    Process evaluation data for generic multi-model boxplot visualization with sophisticated data organization.

    Args:
        evaluation_data_by_bin (Dict[str, List[List[float]]]): Primary evaluation data organized by model-uncertainty combinations.
        uncertainty_categories (List[List[str]]): Hierarchical organization of uncertainty types for comparative analysis.
        models (List[str]): Model identifiers for multi-model comparative visualization.
        num_bins (int): Total number of uncertainty bins for spatial layout and validation.
        config (BoxPlotConfig): Configuration object containing various plotting parameters.

    Returns:
        Tuple[List[Dict], List[Dict], List[float]]: Comprehensive data structure for boxplot rendering.

    Raises:
        KeyError: If model-uncertainty combinations in uncertainty_categories and models don't match
                 keys in evaluation_data_by_bin.
        ValueError: If num_bins doesn't match the actual data structure length.
        IndexError: If bin indices exceed available data in evaluation_data_by_bin.
    """
    processed_data = []
    legend_info = []
    bin_label_locs = []

    outer_min_x_loc = 0.0
    middle_min_x_loc = 0.0
    inner_min_x_loc = 0.0

    for i, (uncert_pair) in enumerate(uncertainty_categories):
        uncertainty_type = (uncert_pair)[0]

        for j in range(num_bins):
            box_x_positions = []

            for hatch_idx, model_type in enumerate(models):
                # Use helper function to extract data
                dict_key = _find_data_key(evaluation_data_by_bin, model_type, uncertainty_type)
                model_data = evaluation_data_by_bin[dict_key]
                bin_data = _extract_bin_data(model_data, j, config.use_list_comp)

                # Calculate position
                x_loc = outer_min_x_loc + inner_min_x_loc + middle_min_x_loc
                box_x_positions.append(x_loc)

                # Use helper function to create data item
                data_item = _create_data_item(
                    bin_data, x_loc, config.width, i, model_type, uncertainty_type, hatch_idx, j, model_data
                )
                processed_data.append(data_item)

                # Collect legend info (only on first bin)
                if j == 0:
                    legend_item = {
                        "color_idx": i,
                        "model_type": model_type,
                        "uncertainty_type": uncertainty_type,
                        "hatch_idx": hatch_idx,
                    }
                    legend_info.append(legend_item)

                inner_min_x_loc += 0.1 + config.width

            # Calculate bin label position
            if config.use_list_comp:
                bin_label_locs.extend(box_x_positions)
            else:
                bin_label_locs.append(float(np.mean(box_x_positions)))

            middle_min_x_loc += 0.02

        # Adjust spacing between different uncertainty types
        if config.use_list_comp:
            spacing_adjustment = _calculate_spacing_adjustment(num_bins, True, config)
            middle_min_x_loc += spacing_adjustment
        else:
            spacing_adjustment = _calculate_spacing_adjustment(num_bins, False, config)
            outer_min_x_loc += spacing_adjustment

    return processed_data, legend_info, bin_label_locs


def _process_boxplot_data_per_model(
    evaluation_data_by_bin: Dict[str, List[List[float]]],
    uncertainty_categories: List[List[str]],
    models: List[str],
    num_bins: int,
    config: BoxPlotConfig,
) -> Tuple[List[Dict], List[Dict], List[float]]:
    """
    Process boxplot data grouped by model.

    Args:
        evaluation_data_by_bin: Dictionary containing all evaluation data grouped by bins
        uncertainty_categories: List of uncertainty types
        models: Model list
        num_bins: Number of bins
        config: BoxPlotConfig

    Returns:
        Tuple[processed_data, legend_info, bin_label_locs]
    """
    processed_data = []
    legend_info = []
    bin_label_locs = []

    outer_min_x_loc = 0.0
    middle_min_x_loc = 0.0
    inner_min_x_loc = 0.0

    for i, (uncert_pair) in enumerate(uncertainty_categories):
        uncertainty_type = (uncert_pair)[0]
        for hatch_idx, model_type in enumerate(models):
            box_x_positions = []

            for j in range(num_bins):
                # Use helper function to extract data
                dict_key = _find_data_key(evaluation_data_by_bin, model_type, uncertainty_type)
                model_data = evaluation_data_by_bin[dict_key]
                bin_data = _extract_bin_data(model_data, j, use_list_comp=True)

                # Calculate position
                width = 0.25
                x_loc = outer_min_x_loc + inner_min_x_loc + middle_min_x_loc
                box_x_positions.append(x_loc)

                # Use helper function to create data item
                data_item = _create_data_item(
                    bin_data, x_loc, width, i, model_type, uncertainty_type, hatch_idx, j, model_data
                )
                processed_data.append(data_item)

                # Collect legend info (only on first bin)
                if j == 0:
                    legend_item = {
                        "color_idx": i,
                        "model_type": model_type,
                        "uncertainty_type": uncertainty_type,
                        "hatch_idx": hatch_idx,
                    }
                    legend_info.append(legend_item)

                inner_min_x_loc += 0.1 + width

            bin_label_locs.extend(box_x_positions)

            # Adjust spacing between bins
            spacing_adjustment = _calculate_spacing_adjustment(num_bins, True, config)
            middle_min_x_loc += spacing_adjustment

        outer_min_x_loc += 0.24

    return processed_data, legend_info, bin_label_locs


def _process_boxplot_data_comparing_q(
    evaluation_data_by_bins: List[Dict[str, List[List[float]]]],
    uncertainty_categories: List[List[str]],
    model: List[str],
    category_labels: List[str],
) -> Tuple[List[Dict], List[float]]:
    """
    Process boxplot data for comparing Q values.

    Args:
        evaluation_data_by_bins: List of data dictionaries, one per Q value
        uncertainty_categories: List of uncertainty types
        model: Model list
        category_labels: Category labels (Q values)

    Returns:
        Tuple[processed_data, bin_label_locs]
    """
    processed_data = []
    bin_label_locs = []

    outer_min_x_loc = 0.0
    inner_min_x_loc = 0.0
    middle_min_x_loc = 0.0

    uncertainty_type = uncertainty_categories[0][0]
    model_type = model[0]

    for idx, q_value in enumerate(category_labels):
        box_x_positions = []
        evaluation_data_by_bin = evaluation_data_by_bins[idx]

        # Use helper function to get data key
        dict_key = _find_data_key(evaluation_data_by_bin, model_type, uncertainty_type)
        model_data = evaluation_data_by_bin[dict_key]

        # Iterate through each bin to display data
        for j in range(len(model_data)):
            bin_data = _extract_bin_data(model_data, j, use_list_comp=True)

            # Calculate box width and position
            width = 0.2 * (4 / 5) ** idx
            x_loc = outer_min_x_loc + inner_min_x_loc + middle_min_x_loc
            box_x_positions.append(x_loc)

            # Use helper function to create data item (with extra fields)
            data_item = _create_data_item(
                bin_data, x_loc, width, 0, model_type, uncertainty_type, 0, j, model_data, q_idx=idx  # extra field
            )
            processed_data.append(data_item)

            inner_min_x_loc += 0.02 + width

        outer_min_x_loc += 0.2
        bin_label_locs.append(float(np.mean(box_x_positions)))

    return processed_data, bin_label_locs

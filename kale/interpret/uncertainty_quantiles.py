# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Wenjie Zhao, mcsoft12138@outlook.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
# =============================================================================

"""
This module implements the uncertainty quantification method from L. A. Schobs, A. J. Swift and H. Lu,
"Uncertainty Estimation for Heatmap-Based Landmark Localization," in IEEE Transactions on Medical Imaging,
vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

Core Classes:
   - QuantileBinningAnalyzer: Main analysis class encapsulating uncertainty quantile analysis
   - QuantileBinningConfig: Configuration for individual bin comparison analysis
   - ComparingBinsConfig: Configuration for comparing different bin counts (Q values)

Analysis Capabilities:
   A) Isotonic regression on uncertainty & error pairs (quantile_binning_and_estimate_errors)
   B) Individual bin comparison: Compare different models/uncertainty types at fixed bin counts
   C) Bin count comparison: Analyze impact of different Q values on model performance
   D) Comprehensive visualization: Boxplots, error bounds, Jaccard metrics, cumulative plots

Example Usage:
    .. code-block:: python

        from kale.interpret.uncertainty_quantiles import QuantileBinningAnalyzer, QuantileBinningConfig

        # Configure analysis
        config = QuantileBinningConfig(
            uncertainty_error_pairs=[('name', 'error_col', 'uncertainty_col')],
            models=['model1', 'model2'],
            dataset='my_dataset',
            target_indices=[0, 1, 2],
            num_bins=10
        )

        # Create analyzer
        analyzer = QuantileBinningAnalyzer(
            config={
                'plot_samples_as_dots': False,
                'show_sample_info': 'detailed',
                'boxplot_error_lim': 64,
                'boxplot_config': {'colormap': 'Set1'},
                'save_folder': 'output/',
                'save_file_preamble': 'analysis_',
                'save_figures': True,
                'interpret': True,
            },
            display_settings={'errors': True, 'jaccard': True, 'error_bounds': True},
        )

        # Run analysis
        analyzer.run_individual_bin_comparison(config)

Dependencies:
   - Box plot configuration classes from kale.interpret.box_plot
   - Uncertainty metrics from kale.evaluate.uncertainty_metrics
   - Similarity metrics from kale.evaluate.similarity_metrics
"""
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import numpy as np

from kale.evaluate.similarity_metrics import evaluate_correlations
from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard, get_mean_errors
from kale.interpret.box_plot import (
    execute_boxplot,
    plot_generic_boxplot,
    plot_per_model_boxplot,
    plot_q_comparing_boxplot,
)
from kale.interpret.uncertainty_utils import plot_cumulative
from kale.prepdata.tabular_transform import generate_struct_for_qbin
from kale.utils.save_xlsx import generate_summary_df

# -----------------------------------------------------------------------------
# Internal Data Dictionary Keys
# -----------------------------------------------------------------------------

# Error data keys
ERROR_DATA_ALL_CONCAT_NOSEP = "all_error_concat_bins_targets_nosep"
ERROR_DATA_ALL_CONCAT_SEP = "all_error_concat_bins_targets_sep_all"
ERROR_DATA_MEAN_BINS_NOSEP = "all_mean_error_bins_nosep"

# Bounds data keys
BOUNDS_DATA_ALL = "error_bounds_all"
BOUNDS_DATA_ALL_CONCAT_SEP = "all_errorbound_concat_bins_targets_sep_all"

# Jaccard data keys
JACCARD_DATA_ALL = "jaccard_all"
JACCARD_DATA_RECALL_ALL = "recall_all"
JACCARD_DATA_PRECISION_ALL = "precision_all"
JACCARD_DATA_ALL_CONCAT_SEP = "all_jaccard_concat_bins_targets_sep_all"

# -----------------------------------------------------------------------------
# Metric Names
# -----------------------------------------------------------------------------
METRIC_NAME_ERROR = "error"
METRIC_NAME_ERRORBOUND = "errorbound"
METRIC_NAME_JACCARD = "jaccard"
METRIC_NAME_RECALL_JACCARD = "recall_jaccard"
METRIC_NAME_PRECISION_JACCARD = "precision_jaccard"
METRIC_NAME_MEAN_ERROR_FOLDS = "mean_error_folds"

# -----------------------------------------------------------------------------
# Display Labels
# -----------------------------------------------------------------------------
LABEL_ERROR_BOUND_ACCURACY = "Error Bound Accuracy (%)"
LABEL_JACCARD_INDEX = "Jaccard Index (%)"
LABEL_LOCALIZATION_ERROR = "Localization Error (mm)"
LABEL_MEAN_ERROR = "Mean Error (mm)"
LABEL_GROUND_TRUTH_BINS_RECALL = "Ground Truth Bins Recall"
LABEL_GROUND_TRUTH_BINS_PRECISION = "Ground Truth Bins Precision"
LABEL_GROUND_TRUTH_BINS_RECALL_PERCENT = "Ground Truth Bins Recall (%)"
LABEL_GROUND_TRUTH_BINS_PRECISION_PERCENT = "Ground Truth Bins Precision (%)"
LABEL_UNCERTAINTY_THRESHOLDED_BIN = "Uncertainty Thresholded Bin"
LABEL_Q_NUM_BINS = "Q (# Bins)"

# -----------------------------------------------------------------------------
# Title and Message Templates
# -----------------------------------------------------------------------------
CUMULATIVE_ERROR_TITLE_TEMPLATE = "Cumulative error for ALL predictions, dataset {}"
CUMULATIVE_ERROR_B1_TITLE_TEMPLATE = "Cumulative error for B1 predictions, dataset {}"
CUMULATIVE_ERROR_B1_VS_ALL_TITLE_TEMPLATE = "{}. Cumulative error comparing ALL and B1, dataset {}"
PLOTTING_TARGET_MESSAGE_TEMPLATE = "Plotting {} for all targets."
PLOTTING_INDIVIDUAL_TARGET_MESSAGE_TEMPLATE = "Plotting individual {} for T{}"

# -----------------------------------------------------------------------------
# File Names
# -----------------------------------------------------------------------------
FILE_NAME_ALL_PREDICTIONS_CUMULATIVE_ERROR = "all_predictions_cumulative_error"
FILE_NAME_B1_PREDICTIONS_CUMULATIVE_ERROR = "b1_predictions_cumulative_error"
FILE_NAME_B1_VS_ALL_CUMULATIVE_ERROR_TEMPLATE = "{}_b1_vs_all_cumulative_error"
FILE_NAME_TARGET_ERRORS = "target_errors"

# Default file format extensions
DEFAULT_FIGURE_FORMAT = "pdf"
DEFAULT_DATA_FORMAT = "xlsx"

# -----------------------------------------------------------------------------
# Summary and Report Strings (for reports and summaries)
# -----------------------------------------------------------------------------
SUMMARY_ALL_TARGETS = "All Targets"
SUMMARY_MEAN_ERROR_TITLE = "Mean error"


@dataclass
class BaseAnalysisConfig:
    """
    Base configuration class with common analysis and visualization settings.

    This base class consolidates shared fields across different analysis configurations to avoid duplication
    and ensure consistency in plotting and analysis parameters.
    """

    # Common analysis settings
    combine_middle_bins: bool = False  # If True, merge middle bins into one for simplified 3-bin analysis
    num_folds: int = 5  # Number of cross-validation folds for statistical robustness
    error_scaling_factor: float = 1.0  # Multiplicative factor for error values (e.g., pixel to mm conversion)

    # Display settings
    show_individual_target_plots: bool = False  # If True, generate separate plots for each target
    individual_targets_to_show: Optional[List[int]] = None  # Target indices for individual plots; use [-1] for all

    # File/save configuration
    save_folder: str = ""  # Directory path where output plots and data files will be saved
    save_file_preamble: str = ""  # Prefix string added to all saved filenames for identification
    save_figures: bool = True  # If True, save plots to disk; if False, display interactively

    # Visualization settings (shared across all analysis types)
    plot_samples_as_dots: bool = False  # If True, overlay individual data points as dots on boxplots
    show_sample_info: Optional[
        str
    ] = None  # Display mode for sample sizes: None (no display), 'text', 'legend', or 'detailed'
    boxplot_error_lim: int = 64  # Upper limit for y-axis on error plots (in mm or pixels)
    percent_y_lim_standard: int = 70  # Y-axis upper limit (%) for standard percentage metrics
    percent_y_lim_extended: int = 120  # Y-axis upper limit (%) for metrics that may exceed 100%
    colormap: str = "Set1"  # Matplotlib colormap name for plot colors (e.g., 'Set1', 'tab10')
    interpret: bool = True  # If True, execute analysis and plotting; if False, skip processing

    def __post_init__(self):
        """
        Initialize default values for mutable fields.

        This method is automatically called after dataclass initialization to set up mutable default values that cannot
        be safely defined in the field declarations.
        """
        if self.individual_targets_to_show is None:
            self.individual_targets_to_show = []


@dataclass
class QuantileBinningConfig(BaseAnalysisConfig):
    """
    Configuration class for quantile binning analysis.

    This class replaces the tuple-based parameter passing with a more structured, type-safe approach for configuring
    quantile binning uncertainty analysis. It inherits common analysis and visualization settings from BaseAnalysisConfig.
    """

    # Data configuration (specific to this analysis type)
    uncertainty_error_pairs: List[Tuple[str, str, str]] = field(
        default_factory=list
    )  # List of (display_name, error_column_name, uncertainty_column_name) tuples for analysis
    models: List[str] = field(default_factory=list)  # List of model names to compare (e.g., ['ResNet50', 'VGG16'])
    dataset: str = ""  # Dataset identifier for labeling and file organization
    target_indices: List[int] = field(default_factory=list)  # List of anatomical landmark/target indices to analyze
    num_bins: int = 10  # Number of uncertainty quantile bins (Q value, typically 5-20)

    # Analysis settings specific to individual bin comparison
    confidence_invert: List[Tuple[str, bool]] = field(
        default_factory=list
    )  # List of (uncertainty_type, should_invert) tuples for proper type safety


@dataclass
class ComparingBinsConfig(BaseAnalysisConfig):
    """
    Configuration class for comparing different bin counts (Q values) analysis.

    This class inherits common analysis and visualization settings from BaseAnalysisConfig, focusing on Q-value
    optimization while reducing code duplication.
    """

    # Data configuration (specific to this analysis type)
    uncertainty_error_pair: Tuple[str, str] = (
        "",
        "",
    )  # Single (uncertainty_type, error_type) tuple to analyze
    model: str = ""  # Single model name to evaluate across different Q values
    dataset: str = ""  # Dataset identifier for labeling and file organization
    targets: List[int] = field(default_factory=list)  # List of anatomical landmark/target indices to analyze
    q_values: List[int] = field(
        default_factory=list
    )  # List of Q values (bin counts) to compare (e.g., [5, 10, 15, 20])
    fitted_save_paths: List[str] = field(default_factory=list)  # Corresponding list of data file paths for each Q value


@dataclass
class MetricPlotConfig:
    """
    Configuration class for metric plotting parameters.

    This class consolidates plotting parameters to reduce function argument count
    and improve maintainability of plotting methods.
    """

    # Core plotting parameters
    eval_data: Dict[str, Any]  # Dictionary containing computed metrics (errors, jaccard, bounds)
    models: List[str]  # List of model names to include in plots
    uncertainty_categories: List[List[str]]  # List of [uncertainty_type, error_type] pairs
    category_labels: List[str]  # X-axis labels (e.g., bin names or Q values)
    num_bins_display: int  # Number of bins to display (may differ from actual if combined)
    plot_func: Callable[..., Any]  # Plotting function (plot_per_model_boxplot, plot_q_comparing_boxplot, etc.)
    show_individual_target_plots: bool  # If True, generate separate plots for each target
    individual_targets_to_show: List[int]  # Target indices for individual plots; use [-1] for all
    detailed_mode: bool  # If True, use enhanced plotting features with additional annotations
    target_indices: Optional[List[int]] = None  # Complete list of target indices for reference

    # Y-axis limits
    percent_y_lim_standard: int = 70  # Y-axis upper limit (%) for standard percentage metrics
    percent_y_lim_extended: int = 120  # Y-axis upper limit (%) for metrics that may exceed 100%

    # Additional plot parameters
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra keyword arguments for plot customization

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.target_indices is None:
            self.target_indices = []


@dataclass
class MetricDefinition:
    """
    Definition of a specific metric's plotting characteristics.

    This class encapsulates the specific parameters needed for each metric type,
    enabling a unified plotting approach across different metrics.
    """

    metric_name: str  # Internal identifier for the metric (e.g., 'error', 'jaccard', 'errorbound')
    data_key: str  # Dictionary key to access metric data in eval_data
    y_label: str  # Y-axis label for plots (e.g., 'Localization Error (mm)')
    to_log: bool = False  # If True, apply logarithmic scaling to y-axis
    convert_to_percent: bool = False  # If True, multiply values by 100 and show as percentages
    y_lim_top_attr: str = "percent_y_lim_standard"  # Attribute name in config to use for y_lim_top
    show_sample_info: Optional[
        str
    ] = None  # Display mode for sample sizes: None (no display), 'text', 'legend', or 'detailed'
    show_individual_dots: bool = False  # If True, overlay individual data points as dots on boxplots
    individual_data_key: Optional[str] = None  # Dictionary key for target-separated data; None if not applicable


class QuantileBinningAnalyzer:
    """
    A class for performing and visualizing Quantile Binning uncertainty analysis.

    This class encapsulates the functionality of the original generate_fig_individual_bin_comparison and
    generate_fig_comparing_bins functions to reduce code duplication and improve maintainability. It provides two core
    analysis modes:
    1. Compare different models/uncertainty types at fixed bin counts (run_individual_bin_comparison).
    2. Compare the impact of different bin counts (Q values) on model performance (run_comparing_bins_analysis).
    """

    # Metric definitions for unified plotting
    _METRIC_DEFINITIONS = {
        "error": [
            MetricDefinition(
                metric_name=METRIC_NAME_ERROR,
                data_key=ERROR_DATA_ALL_CONCAT_NOSEP,
                y_label=LABEL_LOCALIZATION_ERROR,
                to_log=True,
                convert_to_percent=False,
                y_lim_top_attr="boxplot_error_lim",
                show_sample_info="detailed",  # Will be overridden by instance setting
                show_individual_dots=True,  # Will be overridden by instance setting
                individual_data_key=ERROR_DATA_ALL_CONCAT_SEP,
            ),
            MetricDefinition(
                metric_name=METRIC_NAME_MEAN_ERROR_FOLDS,
                data_key=ERROR_DATA_MEAN_BINS_NOSEP,
                y_label=LABEL_MEAN_ERROR,
                to_log=True,
                convert_to_percent=False,
                y_lim_top_attr="boxplot_error_lim",
                show_sample_info=None,
                show_individual_dots=False,
            ),
        ],
        "error_bounds": [
            MetricDefinition(
                metric_name=METRIC_NAME_ERRORBOUND,
                data_key=BOUNDS_DATA_ALL,
                y_label=LABEL_ERROR_BOUND_ACCURACY,
                to_log=False,
                convert_to_percent=True,
                y_lim_top_attr="percent_y_lim_extended",
                show_sample_info=None,
                show_individual_dots=False,
                individual_data_key=BOUNDS_DATA_ALL_CONCAT_SEP,
            )
        ],
        "jaccard": [
            MetricDefinition(
                metric_name=METRIC_NAME_JACCARD,
                data_key=JACCARD_DATA_ALL,
                y_label=LABEL_JACCARD_INDEX,
                to_log=False,
                convert_to_percent=True,
                y_lim_top_attr="percent_y_lim_standard",
                show_sample_info=None,
                show_individual_dots=False,
                individual_data_key=JACCARD_DATA_ALL_CONCAT_SEP,
            ),
            MetricDefinition(
                metric_name=METRIC_NAME_RECALL_JACCARD,
                data_key=JACCARD_DATA_RECALL_ALL,
                y_label=LABEL_GROUND_TRUTH_BINS_RECALL,
                to_log=False,
                convert_to_percent=True,
                y_lim_top_attr="percent_y_lim_extended",
                show_sample_info=None,
                show_individual_dots=False,
            ),
            MetricDefinition(
                metric_name=METRIC_NAME_PRECISION_JACCARD,
                data_key=JACCARD_DATA_PRECISION_ALL,
                y_label=LABEL_GROUND_TRUTH_BINS_PRECISION,
                to_log=False,
                convert_to_percent=True,
                y_lim_top_attr="percent_y_lim_extended",
                show_sample_info=None,
                show_individual_dots=False,
            ),
        ],
    }

    def __init__(
        self,
        config: Dict[str, Any],
        display_settings: Dict[str, Any],
        figure_format: str = DEFAULT_FIGURE_FORMAT,
        data_format: str = DEFAULT_DATA_FORMAT,
    ):
        """
        Initialize the QuantileBinningAnalyzer for uncertainty quantification analysis.

        Args:
            config (Dict[str, Any]): Analyzer configuration dictionary containing plot and save settings:
                - plot_samples_as_dots (bool): Whether to show individual data points as dots on boxplots.
                - show_sample_info (Optional[str]): Mode for displaying sample sizes. Valid values:
                    None, "text", "legend", "detailed".
                - boxplot_error_lim (int): Upper limit for y-axis on error boxplots (typically in mm).
                - boxplot_config (Dict[str, Any]): Configuration for boxplot aesthetics including colormap,
                    font sizes, figure dimensions, and styling parameters.
                - save_folder (str): Folder path where generated plots will be saved.
                - save_file_preamble (str): Prefix string for saved file names to ensure unique identification.
                - save_figures (bool): If True, save plots to disk; if False, display plots interactively.
                - interpret (bool): If True, execute analysis and visualization; if False, skip processing.
            display_settings (Dict[str, bool]): Keys include 'errors', 'error_bounds', 'jaccard',
                'correlation', 'cumulative_error' to control which plots are generated.
            figure_format (str): File extension for figure outputs (e.g., 'pdf', 'png', 'svg').
                Defaults to 'pdf'.
            data_format (str): File extension for data outputs (e.g., 'xlsx', 'csv').
                Defaults to 'xlsx'.
        """
        self.logger = logging.getLogger("qbin")

        plot_config = dict(config)
        plot_config["display_settings"] = display_settings

        self.plot_config = plot_config
        self.display_settings = display_settings
        self.plot_samples_as_dots = config.get("plot_samples_as_dots", False)
        self.show_sample_info = config.get("show_sample_info", None)
        self.boxplot_error_lim = config.get("boxplot_error_lim", 64)
        self.boxplot_config = config.get("boxplot_config", {})
        self.hatch = self.display_settings.get("hatch", "o")
        self.save_folder = config.get("save_folder", "")
        self.save_file_preamble = config.get("save_file_preamble", "")
        self.save_figures = config.get("save_figures", True)
        self.interpret = config.get("interpret", True)
        self.figure_format = figure_format
        self.data_format = data_format

    def _build_filename(self, base_name: str, is_figure: bool = True) -> str:
        """Build full filename with appropriate extension.

        Args:
            base_name (str): Base filename without extension.
            is_figure (bool): If True, use figure_format; otherwise use data_format.

        Returns:
            str: Full filename with extension (e.g., 'plot.pdf' or 'data.xlsx').
        """
        ext = self.figure_format if is_figure else self.data_format
        return f"{base_name}.{ext}"

    def run_individual_bin_comparison(self, config: QuantileBinningConfig) -> None:
        """
        Execute comprehensive comparative analysis of different models and uncertainty types at fixed bin counts.

        This method performs the core individual bin comparison analysis, comparing how different models and
        uncertainty estimation methods perform across uncertainty-based bins. It generates multiple types of
        visualizations and statistical analyses to evaluate uncertainty quantification effectiveness in medical imaging
        applications.

        The analysis workflow includes:
        1. Data loading and evaluation metric computation for all model-uncertainty combinations
        2. Correlation analysis between uncertainty estimates and actual errors (if enabled)
        3. Cumulative error distribution analysis (if enabled)
        4. Error boxplot generation comparing models across uncertainty bins
        5. Error bounds accuracy assessment showing calibration quality
        6. Jaccard similarity analysis evaluating bin overlap with ground truth

        Args:
            config (QuantileBinningConfig): Comprehensive configuration object containing:
                - uncertainty_error_pairs: List of (uncertainty_type, error_type) combinations to analyze
                - models: List of model names to compare (e.g., ['ResNet50', 'VGG16', 'DenseNet'])
                - dataset: Dataset identifier for labeling and file organization
                - target_indices: List of anatomical landmark/target indices to analyze
                - num_bins: Number of uncertainty quantile bins (typically 5-20)
                - combine_middle_bins: Whether to merge middle bins for simplified 3-bin analysis
                - confidence_invert: List of (uncertainty_type, should_invert) tuples for proper type-safe handling
                - Display and visualization settings for plot generation
                - File paths and saving configuration

        Raises:
            FileNotFoundError: If required data files are not found at specified paths.
            ValueError: If configuration parameters are invalid or incompatible.

        Example:

            .. code-block:: pycon

                >>> config = QuantileBinningConfig(
                ...     uncertainty_error_pairs=[
                ...         ("epistemic_uncertainty", "localization_error", "epistemic_uncertainty")
                ...     ],
                ...     models=["ResNet50", "VGG16"],
                ...     dataset="cardiac_mri",
                ...     target_indices=[0, 1, 2, 3],
                ...     num_bins=10,
                ...     combine_middle_bins=False,
                ...     show_individual_target_plots=True,
                ...     individual_targets_to_show=[0, 1],
                ... )
                >>> analyzer.run_individual_bin_comparison(config)

        Note:
            - The method respects the analyzer's display_settings to control which plots are generated
            - Individual target plots provide detailed per-landmark analysis when enabled
            - Results are automatically saved to Excel summary files for further analysis
            - All plots use consistent styling and colormaps for publication-ready figures
        """
        if not self.interpret:
            return

        # Data loading and computation
        # Pass the original 3-tuple format to _gather_evaluation_data
        eval_data = self._gather_evaluation_data(
            config.models,
            config.target_indices,
            config.save_folder,
            config.dataset,
            config.num_bins,
            config.uncertainty_error_pairs,
            config.num_folds,
            config.error_scaling_factor,
            config.combine_middle_bins,
        )

        num_bins_display = 3 if config.combine_middle_bins else config.num_bins
        category_labels = [rf"$B_{{{num_bins_display + 1 - (i + 1)}}}$" for i in range(num_bins_display + 1)]
        # --- Start plotting ---
        if self.display_settings.get("correlation"):
            colormap = self.boxplot_config.get("colormap", "Set1")
            uncertainty_error_model_triples = config.uncertainty_error_pairs
            # confidence_invert is already properly typed as List[Tuple[str, bool]]
            evaluate_correlations(
                eval_data["bins"],
                uncertainty_error_model_triples,
                config.num_bins,
                config.confidence_invert,
                num_folds=config.num_folds,
                colormap=colormap,
                error_scaling_factor=config.error_scaling_factor,
                combine_middle_bins=config.combine_middle_bins,
                save_path=self.save_folder if self.save_figures else None,
                to_log=True,
            )

        if self.display_settings.get("cumulative_error"):
            colormap = self.boxplot_config.get("colormap", "Set1")
            # Extract just the base names (first element) for plot_cumulative
            extracted_uncertainty_error_pairs = [(name, name) for name, err, unc in config.uncertainty_error_pairs]
            plot_cumulative(
                colormap,
                eval_data["bins"],
                config.models,
                extracted_uncertainty_error_pairs,
                np.arange(config.num_bins),
                CUMULATIVE_ERROR_TITLE_TEMPLATE.format(config.dataset),
                save_path=self.save_folder if self.save_figures else None,
                file_name=self._build_filename(FILE_NAME_ALL_PREDICTIONS_CUMULATIVE_ERROR),
                error_scaling_factor=config.error_scaling_factor,
            )
            # Plot cumulative error figure for B1 only predictions
            plot_cumulative(
                colormap,
                eval_data["bins"],
                config.models,
                extracted_uncertainty_error_pairs,
                0,
                CUMULATIVE_ERROR_B1_TITLE_TEMPLATE.format(config.dataset),
                save_path=self.save_folder if self.save_figures else None,
                file_name=self._build_filename(FILE_NAME_B1_PREDICTIONS_CUMULATIVE_ERROR),
                error_scaling_factor=config.error_scaling_factor,
            )

            # Plot cumulative error figure comparing B1 and ALL, for both models
            for model_type in config.models:
                plot_cumulative(
                    colormap,
                    eval_data["bins"],
                    [model_type],
                    extracted_uncertainty_error_pairs,
                    0,
                    CUMULATIVE_ERROR_B1_VS_ALL_TITLE_TEMPLATE.format(model_type, config.dataset),
                    compare_to_all=True,
                    save_path=self.save_folder if self.save_figures else None,
                    file_name=self._build_filename(FILE_NAME_B1_VS_ALL_CUMULATIVE_ERROR_TEMPLATE.format(model_type)),
                    error_scaling_factor=config.error_scaling_factor,
                )

        if self.display_settings.get("errors"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            plot_config = self._create_plot_config(
                eval_data=eval_data,
                models=config.models,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_per_model_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=False,
                target_indices=config.target_indices,
                x_label=LABEL_UNCERTAINTY_THRESHOLDED_BIN,
            )
            self._plot_metrics("error", plot_config)

        if self.display_settings.get("error_bounds"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            plot_config = self._create_plot_config(
                eval_data=eval_data,
                models=config.models,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_generic_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=False,
                target_indices=config.target_indices,
                percent_y_lim_extended=config.percent_y_lim_extended,
                x_label=LABEL_UNCERTAINTY_THRESHOLDED_BIN,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
            )
            self._plot_metrics("error_bounds", plot_config)

        if self.display_settings.get("jaccard"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            plot_config = self._create_plot_config(
                eval_data=eval_data,
                models=config.models,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_generic_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=False,
                target_indices=config.target_indices,
                percent_y_lim_standard=config.percent_y_lim_standard,
                percent_y_lim_extended=config.percent_y_lim_extended,
                x_label=LABEL_UNCERTAINTY_THRESHOLDED_BIN,
                recall_y_label=LABEL_GROUND_TRUTH_BINS_RECALL,
                precision_y_label=LABEL_GROUND_TRUTH_BINS_PRECISION,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
            )
            self._plot_metrics("jaccard", plot_config)

    def run_comparing_bins_analysis(self, config: ComparingBinsConfig) -> None:
        """
        Execute comprehensive analysis comparing the impact of different bin counts (Q values) on model performance.

        This method performs Q-value optimization analysis to determine the optimal number of uncertainty bins for a
        specific model and uncertainty type combination. It systematically evaluates how different binning strategies
        (Q=5, Q=10, Q=15, etc.) affect uncertainty quantification performance and provides insights for hyperparameter
        selection in medical imaging applications.

        The analysis workflow includes:
        1. Iterative data collection across all specified Q values
        2. Metric computation (errors, Jaccard similarity, error bounds) for each Q configuration
        3. Comparative visualization showing Q-value impact on performance metrics
        4. Statistical analysis of optimal bin count selection
        5. Individual target analysis (if enabled) for detailed per-landmark insights

        This analysis is crucial for:
        - Determining optimal binning strategies for specific datasets and models
        - Understanding the trade-offs between granularity and statistical reliability
        - Identifying Q values that maximize uncertainty-error correlation
        - Validating binning approach generalizability across different anatomical targets

        Args:
            config (ComparingBinsConfig): Comprehensive configuration object containing:
                - uncertainty_error_pair: Single (uncertainty_type, error_type) tuple to analyze
                - model: Single model name to evaluate across different Q values
                - dataset: Dataset identifier for consistent labeling and organization
                - targets: List of target indices for multi-target analysis
                - q_values: List of Q values to compare (e.g., [5, 10, 15, 20, 25])
                - fitted_save_paths: Corresponding list of data file paths for each Q value
                - combine_middle_bins: Whether to use simplified 3-bin analysis for all Q values
                - Display settings controlling which metrics to visualize
                - Individual target plotting configuration for detailed analysis

        Raises:
            FileNotFoundError: If data files for any Q value are missing or inaccessible.
            ValueError: If Q values and save paths lists have mismatched lengths.
            IndexError: If specified target indices are not present in the data.

        Example:

            .. code-block:: pycon

                >>> config = ComparingBinsConfig(
                ...     uncertainty_error_pair=("epistemic_uncertainty", "localization_error"),
                ...     model="ResNet50",
                ...     dataset="cardiac_mri",
                ...     targets=[0, 1, 2, 3],
                ...     q_values=[5, 10, 15, 20],
                ...     fitted_save_paths=[
                ...         "data/q5_results/",
                ...         "data/q10_results/",
                ...         "data/q15_results/",
                ...         "data/q20_results/",
                ...     ],
                ...     show_individual_target_plots=True,
                ...     individual_targets_to_show=[0],
                ... )
                >>> analyzer.run_comparing_bins_analysis(config)

        Note:
            - Results typically show optimal Q values in the range of 10-20 bins for medical imaging
            - Higher Q values provide finer granularity but may suffer from insufficient statistics
            - Lower Q values are more robust but may miss important uncertainty patterns
            - Individual target analysis can reveal target-specific optimal Q values
            - Generated plots use consistent formatting for easy comparison across Q values
        """
        if not self.interpret:
            return

        model_list = [config.model]
        # Convert single 2-tuple (uncertainty_type, error_type) to 3-tuple format for _gather_evaluation_data
        # Add uncertainty_type as the third element (display_name = uncertainty_type for consistency)
        uncertainty_error_pair_list = [
            (config.uncertainty_error_pair[0], config.uncertainty_error_pair[1], config.uncertainty_error_pair[0])
        ]

        # Collect data for each Q value
        all_eval_data_by_q: Dict[str, List[Any]] = {
            ERROR_DATA_ALL_CONCAT_NOSEP: [],
            ERROR_DATA_ALL_CONCAT_SEP: [],
            ERROR_DATA_MEAN_BINS_NOSEP: [],
            BOUNDS_DATA_ALL: [],
            BOUNDS_DATA_ALL_CONCAT_SEP: [],
            JACCARD_DATA_ALL: [],
            JACCARD_DATA_RECALL_ALL: [],
            JACCARD_DATA_PRECISION_ALL: [],
            JACCARD_DATA_ALL_CONCAT_SEP: [],
        }
        for idx, num_bins in enumerate(config.q_values):
            eval_data = self._gather_evaluation_data(
                model_list,
                config.targets,
                config.fitted_save_paths[idx],
                config.dataset,
                num_bins,
                uncertainty_error_pair_list,
                config.num_folds,
                config.error_scaling_factor,
                config.combine_middle_bins,
            )
            all_eval_data_by_q[ERROR_DATA_ALL_CONCAT_NOSEP].append(eval_data["errors"][ERROR_DATA_ALL_CONCAT_NOSEP])
            all_eval_data_by_q[ERROR_DATA_ALL_CONCAT_SEP].append(eval_data["errors"][ERROR_DATA_ALL_CONCAT_SEP])
            all_eval_data_by_q[ERROR_DATA_MEAN_BINS_NOSEP].append(eval_data["errors"][ERROR_DATA_MEAN_BINS_NOSEP])
            all_eval_data_by_q[BOUNDS_DATA_ALL].append(eval_data["bounds"][BOUNDS_DATA_ALL])
            all_eval_data_by_q[BOUNDS_DATA_ALL_CONCAT_SEP].append(eval_data["bounds"][BOUNDS_DATA_ALL_CONCAT_SEP])
            all_eval_data_by_q[JACCARD_DATA_ALL].append(eval_data["jaccard"][JACCARD_DATA_ALL])
            all_eval_data_by_q[JACCARD_DATA_RECALL_ALL].append(eval_data["jaccard"][JACCARD_DATA_RECALL_ALL])
            all_eval_data_by_q[JACCARD_DATA_PRECISION_ALL].append(eval_data["jaccard"][JACCARD_DATA_PRECISION_ALL])
            all_eval_data_by_q[JACCARD_DATA_ALL_CONCAT_SEP].append(eval_data["jaccard"][JACCARD_DATA_ALL_CONCAT_SEP])

        # num_bins_display uses the final Q value from the loop (last element of q_values) for display configuration
        num_bins_display = 3 if config.combine_middle_bins else num_bins
        category_labels = [str(q) for q in config.q_values]

        # --- Start plotting ---
        # Convert data structure to match the helper method expectations
        combined_eval_data = {
            "errors": {
                ERROR_DATA_ALL_CONCAT_NOSEP: all_eval_data_by_q[ERROR_DATA_ALL_CONCAT_NOSEP],
                ERROR_DATA_ALL_CONCAT_SEP: all_eval_data_by_q[ERROR_DATA_ALL_CONCAT_SEP],
                ERROR_DATA_MEAN_BINS_NOSEP: all_eval_data_by_q[ERROR_DATA_MEAN_BINS_NOSEP],
            },
            "bounds": {
                BOUNDS_DATA_ALL: all_eval_data_by_q[BOUNDS_DATA_ALL],
                BOUNDS_DATA_ALL_CONCAT_SEP: all_eval_data_by_q[BOUNDS_DATA_ALL_CONCAT_SEP],
            },
            "jaccard": {
                JACCARD_DATA_ALL: all_eval_data_by_q[JACCARD_DATA_ALL],
                JACCARD_DATA_RECALL_ALL: all_eval_data_by_q[JACCARD_DATA_RECALL_ALL],
                JACCARD_DATA_PRECISION_ALL: all_eval_data_by_q[JACCARD_DATA_PRECISION_ALL],
                JACCARD_DATA_ALL_CONCAT_SEP: all_eval_data_by_q[JACCARD_DATA_ALL_CONCAT_SEP],
            },
        }

        if self.display_settings.get("errors"):
            uncertainty_categories = [[config.uncertainty_error_pair[0], config.uncertainty_error_pair[1]]]
            plot_config = self._create_plot_config(
                eval_data=combined_eval_data,
                models=model_list,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_q_comparing_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=True,
                target_indices=config.targets,
                x_label=LABEL_Q_NUM_BINS,
            )
            self._plot_metrics("error", plot_config)

        if self.display_settings.get("error_bounds"):
            uncertainty_categories = [[config.uncertainty_error_pair[0], config.uncertainty_error_pair[1]]]
            plot_config = self._create_plot_config(
                eval_data=combined_eval_data,
                models=model_list,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_q_comparing_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=True,
                target_indices=config.targets,
                percent_y_lim_extended=config.percent_y_lim_extended,
                x_label=LABEL_Q_NUM_BINS,
            )
            self._plot_metrics("error_bounds", plot_config)

        if self.display_settings.get("jaccard"):
            uncertainty_categories = [[config.uncertainty_error_pair[0], config.uncertainty_error_pair[1]]]
            plot_config = self._create_plot_config(
                eval_data=combined_eval_data,
                models=model_list,
                uncertainty_categories=uncertainty_categories,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                plot_func=plot_q_comparing_boxplot,
                show_individual_target_plots=config.show_individual_target_plots,
                individual_targets_to_show=config.individual_targets_to_show or [],
                detailed_mode=True,
                target_indices=config.targets,
                percent_y_lim_standard=config.percent_y_lim_standard,
                percent_y_lim_extended=config.percent_y_lim_extended,
                x_label=LABEL_Q_NUM_BINS,
                recall_y_label=LABEL_GROUND_TRUTH_BINS_RECALL_PERCENT,
                precision_y_label=LABEL_GROUND_TRUTH_BINS_PRECISION_PERCENT,
            )
            self._plot_metrics("jaccard", plot_config)

    def _get_save_location(self, suffix: str, show_individual_dots: bool) -> Optional[str]:
        """
        Construct and return the save path for charts, or return None if not saving.

        This method generates the complete file path for saving plots based on the analyzer's configuration and the
        specific plot parameters.

        Args:
            suffix (str): Descriptive suffix for the filename (e.g., "error_all_targets", "jaccard_target_1").
            show_individual_dots (bool): Whether individual data points are shown as dots on the plot.
                This affects the filename to distinguish between dotted and undotted variants.

        Returns:
            Optional[str]: Complete file path with .pdf extension if saving is enabled,
                          None if save_figures is False.
        """
        if not self.save_figures:
            return None

        filename_suffix = "_dotted" if show_individual_dots else "_undotted"
        return os.path.join(self.save_folder, f"{self.save_file_preamble}{filename_suffix}_{suffix}.pdf")

    def _gather_evaluation_data(
        self,
        models: List[str],
        targets: List[int],
        save_path_pre: str,
        dataset: str,
        num_bins: int,
        uncertainty_pairs: List[Tuple[str, str, str]],
        num_folds: int,
        error_scaling_factor: float,
        combine_middle_bins: bool,
    ) -> Dict[str, Any]:
        """
        Load and compute all evaluation metrics for the given binning configuration.

        Args:
            models (List[str]): List of model names.
            targets (List[int]): List of target indices.
            save_path_pre (str): Path prefix for loading precomputed binning data.
            dataset (str): Dataset name.
            num_bins (int): Number of bins.
            uncertainty_pairs (List[Tuple[str, str, str]]): Uncertainty-error pair configurations in 3-tuple format.
            num_folds (int): Number of cross-validation folds.
            error_scaling_factor (float): Error scaling factor.
            combine_middle_bins (bool): Whether to combine middle bins.

        Returns:
            Dict[str, Any]: Dictionary containing all computed metrics (errors, Jaccard, bounds, etc.).
        """
        bins_all_targets, _, bounds_all_targets, _ = generate_struct_for_qbin(models, targets, save_path_pre, dataset)

        error_data = get_mean_errors(
            bins_all_targets, uncertainty_pairs, num_bins, targets, num_folds, error_scaling_factor, combine_middle_bins
        )
        jaccard_data = evaluate_jaccard(
            bins_all_targets, uncertainty_pairs, num_bins, targets, num_folds, combine_middle_bins
        )
        bounds_data = evaluate_bounds(
            bounds_all_targets, bins_all_targets, uncertainty_pairs, num_bins, targets, num_folds, combine_middle_bins
        )
        generate_summary_df(
            error_data,
            [[ERROR_DATA_MEAN_BINS_NOSEP, SUMMARY_ALL_TARGETS]],
            SUMMARY_MEAN_ERROR_TITLE,
            os.path.join(self.save_folder, self._build_filename(FILE_NAME_TARGET_ERRORS, is_figure=False)),
        )
        return {"errors": error_data, "jaccard": jaccard_data, "bounds": bounds_data, "bins": bins_all_targets}

    def _plot_metrics(self, metric_type: str, config: MetricPlotConfig) -> None:
        """
        Unified method for plotting different types of metrics.

        This method replaces the three similar _plot_*_metrics methods with a single,
        configurable approach that reduces code duplication.

        Args:
            metric_type (str): Type of metric ('error', 'error_bounds', 'jaccard')
            config (MetricPlotConfig): Configuration object containing all plotting parameters
        """
        if metric_type not in self._METRIC_DEFINITIONS:
            raise ValueError(f"Unknown metric type: {metric_type}. Available: {list(self._METRIC_DEFINITIONS.keys())}")

        metric_definitions = copy.deepcopy(self._METRIC_DEFINITIONS[metric_type])

        # Handle special case for jaccard metrics with custom y_labels from kwargs
        if metric_type == "jaccard" and config.kwargs:
            recall_y_label = config.kwargs.get("recall_y_label", LABEL_GROUND_TRUTH_BINS_RECALL)
            precision_y_label = config.kwargs.get("precision_y_label", LABEL_GROUND_TRUTH_BINS_PRECISION)

            # Update metric definitions with custom labels
            for metric_def in metric_definitions:
                if metric_def.metric_name == METRIC_NAME_RECALL_JACCARD:
                    metric_def.y_label = recall_y_label
                elif metric_def.metric_name == METRIC_NAME_PRECISION_JACCARD:
                    metric_def.y_label = precision_y_label

        # Get the appropriate data section
        if metric_type == "error":
            data_section = config.eval_data["errors"]
        elif metric_type == "error_bounds":
            data_section = config.eval_data["bounds"]
        elif metric_type == "jaccard":
            data_section = config.eval_data["jaccard"]

        # Plot each metric definition
        for metric_def in metric_definitions:
            # Skip mean error folds for individual target plots
            if metric_def.metric_name == METRIC_NAME_MEAN_ERROR_FOLDS and config.show_individual_target_plots:
                continue

            # Determine y_lim_top from config
            if metric_def.y_lim_top_attr == "boxplot_error_lim":
                y_lim_top = self.boxplot_error_lim
            elif metric_def.y_lim_top_attr == "percent_y_lim_standard":
                y_lim_top = config.percent_y_lim_standard
            elif metric_def.y_lim_top_attr == "percent_y_lim_extended":
                y_lim_top = config.percent_y_lim_extended
            else:
                y_lim_top = None

            # Override with instance settings for sample info and dots
            show_sample_info = (
                self.show_sample_info if metric_def.metric_name == METRIC_NAME_ERROR else metric_def.show_sample_info
            )
            show_individual_dots = (
                self.plot_samples_as_dots
                if metric_def.metric_name == METRIC_NAME_ERROR
                else metric_def.show_individual_dots
            )

            # Prepare kwargs for this specific metric
            metric_kwargs = config.kwargs.copy() if config.kwargs else {}
            if "y_label" not in metric_kwargs:
                metric_kwargs["y_label"] = metric_def.y_label

            # Plot main metric
            self._plot_metric(
                metric_def.metric_name,
                config.plot_func,
                data_section[metric_def.data_key],
                config.models,
                config.uncertainty_categories,
                config.category_labels,
                show_sample_info=show_sample_info,
                to_log=metric_def.to_log,
                convert_to_percent=metric_def.convert_to_percent,
                num_bins_display=config.num_bins_display,
                show_individual_dots=show_individual_dots,
                y_lim_top=y_lim_top,
                detailed_mode=config.detailed_mode,
                **metric_kwargs,
            )

            # Plot individual targets if requested and data available
            if (
                config.show_individual_target_plots
                and metric_def.individual_data_key
                and config.individual_targets_to_show
            ):
                # Prepare kwargs for individual targets, ensuring no duplicate y_label
                individual_kwargs = metric_kwargs.copy()
                individual_kwargs.update(
                    {
                        "show_sample_info": show_sample_info,
                        "y_label": metric_def.y_label,
                        "to_log": metric_def.to_log,
                        "convert_to_percent": metric_def.convert_to_percent,
                        "show_individual_dots": show_individual_dots,
                        "y_lim_top": y_lim_top,
                        "target_indices": config.target_indices,
                        "detailed_mode": config.detailed_mode,
                    }
                )

                self._plot_individual_targets(
                    metric_def.metric_name,
                    config.plot_func,
                    data_section[metric_def.individual_data_key],
                    config.individual_targets_to_show,
                    config.uncertainty_categories,
                    config.models,
                    config.category_labels,
                    config.num_bins_display,
                    **individual_kwargs,
                )

    def _plot_individual_targets(
        self,
        metric_name: str,
        plot_func: Callable[..., Any],
        sep_target_data: List[Any],
        individual_targets_to_show: List[int],
        uncertainty_categories: List[List[str]],
        models: List[str],
        category_labels: List[str],
        num_bins_display: int,
        show_individual_dots: bool,
        target_indices: Optional[List[int]],
        detailed_mode: bool,
        **kwargs: Any,
    ):
        """
        Generate individual target-specific plots for detailed analysis.

        This method creates separate plots for each specified target to enable detailed analysis of model performance
        on individual anatomical landmarks or regions. It handles different data structures depending on the plotting
        mode.

        Args:
            metric_name (str): Name of the metric being plotted (e.g., "error", "jaccard", "errorbound").
            plot_func (Callable): Plotting function to use (plot_per_model_boxplot, plot_q_comparing_boxplot, etc.).
            sep_target_data (List[Any]): Data separated by target. Structure varies by plotting mode:
                - For individual bin comparison: List of target data dictionaries
                - For comparing Q: List of Q-value data with target indices
            individual_targets_to_show (List[int]): List of target indices to plot individually.
                Use [-1] to plot all targets.
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations.
            models (List[str]): List of model names to analyze.
            category_labels (List[str]): Labels for x-axis categories.
            num_bins_display (int): Number of bins to display on the plot.
            show_individual_dots (bool): Whether to show individual data points as dots.
            target_indices (Optional[List[int]]): Complete list of target indices for reference.
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced features.
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        individual_targets_data = []
        if plot_func == plot_q_comparing_boxplot:
            # This logic is for comparing_q mode where sep_target_data is structured differently.
            if target_indices is not None:
                for t_idx in target_indices:
                    if t_idx in individual_targets_to_show or individual_targets_to_show == [-1]:
                        # Extract data for this target across all Q values
                        # Map target ID to its position in the targets list
                        target_pos = target_indices.index(t_idx)
                        target_data_across_q = [q_data[target_pos] for q_data in sep_target_data]
                        individual_targets_data.append({"target_idx": t_idx, "data": target_data_across_q})
        else:
            # This logic is for individual bin comparison mode.
            for idx, target_data in enumerate(sep_target_data):
                if idx in individual_targets_to_show or individual_targets_to_show == [-1]:
                    individual_targets_data.append({"target_idx": idx, "data": [target_data]})

        for target_info in individual_targets_data:
            target_idx = cast(int, target_info["target_idx"])
            target_data = cast(List[Dict[str, List[List[float]]]], target_info["data"])
            self.logger.info(PLOTTING_INDIVIDUAL_TARGET_MESSAGE_TEMPLATE.format(metric_name, target_idx))

            execute_boxplot(
                plot_func=plot_func,
                evaluation_data_by_bins=target_data,
                uncertainty_categories=uncertainty_categories,
                models=models,
                category_labels=category_labels,
                num_bins_display=num_bins_display,
                save_path=self._get_save_location(f"{metric_name}_target_{target_idx}", show_individual_dots),
                boxplot_config_base=self.boxplot_config,
                show_individual_dots=show_individual_dots,
                detailed_mode=detailed_mode,
                **kwargs,
            )

    def _plot_metric(
        self,
        metric_name: str,
        plot_func: Callable[..., Any],
        all_targets_data: List[Dict],
        models: List[str],
        uncertainty_categories: List[List[str]],
        category_labels: List[str],
        show_sample_info: Optional[str],
        x_label: str,
        y_label: str,
        to_log: bool,
        convert_to_percent: bool,
        num_bins_display: int,
        show_individual_dots: bool,
        detailed_mode: bool,
        **kwargs: Any,
    ):
        """
        Universal metric plotting function for drawing boxplots of various uncertainty quantification metrics.

        This is the core plotting method that handles the creation of boxplots for different metrics including errors,
        Jaccard index, and error bounds. It standardizes the plotting process across different metric types and
        plotting modes.

        Args:
            metric_name (str): The metric being plotted ('error', 'jaccard', 'errorbound', 'mean_error_folds',
                'recall_jaccard', 'precision_jaccard').
            plot_func (Callable): The plotting function to use (plot_generic_boxplot, plot_per_model_boxplot,
                plot_q_comparing_boxplot).
            all_targets_data (List[Dict]): Raw data organized for all targets. Structure varies:
                - For individual bin comparison: Single dictionary or list with one dictionary
                - For comparing Q plots: List of dictionaries (one per Q value)
            models (List[str]): List of model names to analyze.
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations. Each inner list
                contains [uncertainty_type, error_type].
            category_labels (List[str]): Labels for x-axis categories (bins, Q values, etc.).
            show_sample_info (str): Mode for displaying sample size information ("None", "text", "legend").
            x_label (str): Label for the x-axis (e.g., "Uncertainty Thresholded Bin", "Q (# Bins)").
            y_label (str): Label for the y-axis (e.g., "Localization Error (mm)", "Jaccard Index (%)").
            to_log (bool): Whether to use logarithmic scaling for the y-axis.
            convert_to_percent (bool): Whether to convert values to percentages for display.
            num_bins_display (int): Number of bins to display on the plot.
            show_individual_dots (bool): Whether to show individual data points as dots on boxplots.
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced features.
            **kwargs: Additional keyword arguments passed to the boxplot configuration including y_lim_top,
                y_lim_bottom, font_size_label, font_size_tick, width, colormap.
        """
        # Plot aggregated charts for all targets
        self.logger.info(PLOTTING_TARGET_MESSAGE_TEMPLATE.format(metric_name))

        # Handle different data structures for different plot functions
        if plot_func == plot_q_comparing_boxplot:
            # For comparing Q plots, data is a list of dictionaries
            plot_data_all = all_targets_data
        else:
            # For individual bin comparison, check if data is already a list
            if isinstance(all_targets_data, list):
                plot_data_all = all_targets_data
            else:
                plot_data_all = [all_targets_data]

        execute_boxplot(
            plot_func=plot_func,
            evaluation_data_by_bins=plot_data_all,
            uncertainty_categories=uncertainty_categories,
            models=models,
            category_labels=category_labels,
            num_bins_display=num_bins_display,
            save_path=self._get_save_location(f"{metric_name}_all_targets", show_individual_dots),
            boxplot_config_base=self.boxplot_config,
            show_individual_dots=show_individual_dots,
            detailed_mode=detailed_mode,
            x_label=x_label,
            y_label=y_label,
            to_log=to_log,
            convert_to_percent=convert_to_percent,
            show_sample_info=show_sample_info,
            **kwargs,
        )

    def _create_plot_config(
        self,
        eval_data: Dict[str, Any],
        models: List[str],
        uncertainty_categories: List[List[str]],
        category_labels: List[str],
        num_bins_display: int,
        plot_func: Callable[..., Any],
        show_individual_target_plots: bool,
        individual_targets_to_show: List[int],
        detailed_mode: bool,
        target_indices: Optional[List[int]] = None,
        percent_y_lim_standard: int = 70,
        percent_y_lim_extended: int = 120,
        **kwargs: Any,
    ) -> MetricPlotConfig:
        """
        Create a MetricPlotConfig object from the provided parameters.

        This helper method reduces the number of arguments passed to plotting methods by encapsulating them in a
        configuration object. It serves as a factory method to construct the MetricPlotConfig dataclass with
        consistent parameter organization and default value handling.

        Args:
            eval_data (Dict[str, Any]): Comprehensive evaluation data dictionary containing computed metrics.
                Structure: {'errors': {...}, 'bounds': {...}, 'jaccard': {...}, 'bins': {...}}.
                Each metric category contains aggregated and separated data for all models and targets.
            models (List[str]): List of model names to include in the plots (e.g., ['ResNet50', 'VGG16']).
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations for plotting.
                Each inner list contains [uncertainty_type, error_type] (e.g., [['epistemic', 'localization']]).
            category_labels (List[str]): Labels for x-axis categories. Can be bin labels (e.g., ['B_10', 'B_9', ...])
                or Q values (e.g., ['5', '10', '15', '20']).
            num_bins_display (int): Number of bins to display in the plot. May differ from actual bin count if
                combine_middle_bins is True (typically 3 for combined, 5-20 for uncombined).
            plot_func (Callable[..., Any]): Plotting function to use for visualization. Options include:
                - plot_per_model_boxplot: For individual bin comparison mode
                - plot_q_comparing_boxplot: For Q-value comparison mode
                - plot_generic_boxplot: For generic metric visualization
            show_individual_target_plots (bool): Whether to generate separate plots for individual anatomical targets.
            individual_targets_to_show (List[int]): List of target indices for which to generate individual plots.
                Use [-1] to plot all targets, or specify indices (e.g., [0, 1, 2]).
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced visualization features.
                Set to False for individual bin comparison, True for comparing Q analysis.
            target_indices (Optional[List[int]], optional): Complete list of all target indices in the dataset.
                Required when plotting individual targets to maintain proper indexing. Defaults to None.
            percent_y_lim_standard (int, optional): Standard upper limit for percentage-based y-axis (0-100 scale).
                Used for Jaccard Index plots. Defaults to 70.
            percent_y_lim_extended (int, optional): Extended upper limit for percentage-based y-axis.
                Used for error bounds and recall/precision plots where values may exceed 100%. Defaults to 120.
            **kwargs (Any): Additional keyword arguments passed to the plotting functions. Common kwargs include:
                - x_label (str): Custom x-axis label
                - y_label (str): Custom y-axis label
                - width (float): Bar width for boxplots
                - y_lim_bottom (int): Lower y-axis limit
                - font_size_label (int): Font size for axis labels
                - font_size_tick (int): Font size for tick labels
                - recall_y_label (str): Custom label for Jaccard recall plots
                - precision_y_label (str): Custom label for Jaccard precision plots

        Returns:
            MetricPlotConfig: Configuration object encapsulating all plotting parameters in a structured format.
                This object is passed to _plot_metrics() to generate the actual visualizations.
        """
        return MetricPlotConfig(
            eval_data=eval_data,
            models=models,
            uncertainty_categories=uncertainty_categories,
            category_labels=category_labels,
            num_bins_display=num_bins_display,
            plot_func=plot_func,
            show_individual_target_plots=show_individual_target_plots,
            individual_targets_to_show=individual_targets_to_show,
            detailed_mode=detailed_mode,
            target_indices=target_indices,
            percent_y_lim_standard=percent_y_lim_standard,
            percent_y_lim_extended=percent_y_lim_extended,
            kwargs=kwargs,
        )

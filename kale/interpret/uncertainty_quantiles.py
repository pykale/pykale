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
   A) Isotonic regression on uncertainty & error pairs (quantile_binning_and_est_errors)
   B) Individual bin comparison: Compare different models/uncertainty types at fixed bin counts
   C) Bin count comparison: Analyze impact of different Q values on model performance
   D) Comprehensive visualization: Boxplots, error bounds, Jaccard metrics, cumulative plots

Example Usage:
    ```python
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
        display_settings={'errors': True, 'jaccard': True, 'error_bounds': True},
        save_folder='output/',
        save_file_preamble='analysis_',
        save_figures=True,
        interpret=True,
        samples_as_dots_bool=False,
        show_sample_info='detailed',
        box_plot_error_lim=64,
        boxplot_config={'colormap': 'Set1'}
    )

    # Run analysis
    analyzer.individual_bin_comparison(config, detailed_mode=True)
    ```

Dependencies:
   - Box plot configuration classes from kale.interpret.box_plot
   - Uncertainty metrics from kale.evaluate.uncertainty_metrics
   - Similarity metrics from kale.evaluate.similarity_metrics
"""
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np

from kale.evaluate.similarity_metrics import evaluate_correlations
from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard, get_mean_errors
from kale.interpret.box_plot import (
    execute_boxplot,
    plot_comparing_q_boxplot,
    plot_generic_boxplot,
    plot_per_model_boxplot,
)
from kale.interpret.uncertainty_utils import plot_cumulative
from kale.prepdata.tabular_transform import generate_struct_for_qbin
from kale.utils.save_xlsx import generate_summary_df

# Constants for hardcoded strings

# Error data keys
ERROR_DATA_ALL_CONCAT_NOSEP = "all error concat bins targets nosep"
ERROR_DATA_ALL_CONCAT_SEP = "all error concat bins targets sep all"
ERROR_DATA_MEAN_BINS_NOSEP = "all mean error bins nosep"

# Bounds data keys
BOUNDS_DATA_ALL = "Error Bounds All"
BOUNDS_DATA_ALL_CONCAT_SEP = "all errorbound concat bins targets sep all"

# Jaccard data keys
JACCARD_DATA_ALL = "Jaccard All"
JACCARD_DATA_RECALL_ALL = "Recall All"
JACCARD_DATA_PRECISION_ALL = "Precision All"
JACCARD_DATA_ALL_CONCAT_SEP = "all jacc concat bins targets sep all"

# Labels and display text
LABEL_ERROR_BOUND_ACCURACY = "Error Bound Accuracy (%)"
LABEL_JACCARD_INDEX = "Jaccard Index (%)"
LABEL_LOCALIZATION_ERROR = "Localization Error (mm)"
LABEL_MEAN_ERROR = "Mean Error (mm)"
LABEL_GROUND_TRUTH_BINS_RECALL = "Ground Truth Bins Recall"
LABEL_GROUND_TRUTH_BINS_PRECISION = "Ground Truth Bins Precision"
LABEL_UNCERTAINTY_THRESHOLDED_BIN = "Uncertainty Thresholded Bin"
LABEL_Q_NUM_BINS = "Q (# Bins)"

# Metric names for plotting
METRIC_NAME_ERROR = "error"
METRIC_NAME_ERRORBOUND = "errorbound"
METRIC_NAME_JACCARD = "jaccard"
METRIC_NAME_RECALL_JACCARD = "recall_jaccard"
METRIC_NAME_PRECISION_JACCARD = "precision_jaccard"
METRIC_NAME_MEAN_ERROR_FOLDS = "mean_error_folds"

# Template strings
CUMULATIVE_ERROR_TITLE_TEMPLATE = "Cumulative error for ALL predictions, dataset {}"
PLOTTING_TARGET_MESSAGE_TEMPLATE = "Plotting {} for all targets."
PLOTTING_INDIVIDUAL_TARGET_MESSAGE_TEMPLATE = "Plotting individual {} for T{}"

# Summary and report strings
SUMMARY_ALL_TARGETS = "All Targets"
SUMMARY_MEAN_ERROR_TITLE = "Mean error"


@dataclass
class QuantileBinningConfig:
    """
    Configuration class for quantile binning analysis.

    This class replaces the tuple-based parameter passing with a more structured, type-safe approach for configuring
    quantile binning uncertainty analysis.
    """

    # Data configuration
    uncertainty_error_pairs: List[Tuple[str, str, str]]
    models: List[str]
    dataset: str
    target_indices: List[int]
    num_bins: int

    # Analysis settings
    combine_middle_bins: bool = False
    confidence_invert: Optional[List[List[Union[str, bool]]]] = None
    num_folds: int = 5
    error_scaling_factor: float = 1.0

    # Display settings
    show_individual_target_plots: bool = False
    ind_targets_to_show: Optional[List[int]] = None

    # File/save configuration
    save_folder: str = ""
    save_file_preamble: str = ""
    save_figures: bool = True

    # Visualization settings
    samples_as_dots_bool: bool = False
    show_sample_info: str = "None"
    box_plot_error_lim: int = 64
    percent_y_lim_standard: int = 70
    percent_y_lim_extended: int = 120
    colormap: str = "Set1"
    interpret: bool = True

    def __post_init__(self):
        """
        Initialize default values for mutable fields.

        This method is automatically called after dataclass initialization to set up mutable default values that cannot
        be safely defined in the field declarations.
        """
        if self.ind_targets_to_show is None:
            self.ind_targets_to_show = []


@dataclass
class ComparingBinsConfig:
    """
    Configuration class for comparing different bin counts (Q values) analysis.
    """

    # Data configuration
    uncertainty_error_pair: Tuple[str, str, str]
    model: str
    dataset: str
    targets: List[int]
    all_values_q: List[int]
    all_fitted_save_paths: List[str]

    # Analysis settings
    combine_middle_bins: bool = False
    num_folds: int = 5
    error_scaling_factor: float = 1.0

    # Display settings
    show_individual_target_plots: bool = False
    ind_targets_to_show: Optional[List[int]] = None

    # File/save configuration
    save_folder: str = ""
    save_file_preamble: str = ""
    save_figures: bool = True

    # Visualization settings
    samples_as_dots_bool: bool = False
    show_sample_info: str = "None"
    box_plot_error_lim: int = 64
    percent_y_lim_standard: int = 70
    percent_y_lim_extended: int = 120
    colormap: str = "Set1"
    interpret: bool = True

    def __post_init__(self):
        """
        Initialize default values for mutable fields.

        This method is automatically called after dataclass initialization to set up mutable default values that cannot
        be safely defined in the field declarations.
        """
        if self.ind_targets_to_show is None:
            self.ind_targets_to_show = []


class QuantileBinningAnalyzer:
    """
    A class for performing and visualizing Quantile Binning uncertainty analysis.

    This class encapsulates the functionality of the original generate_fig_individual_bin_comparison and
    generate_fig_comparing_bins functions to reduce code duplication and improve maintainability. It provides two core
    analysis modes:
    1. Compare different models/uncertainty types at fixed bin counts (individual_bin_comparison).
    2. Compare the impact of different bin counts (Q values) on model performance (comparing_bins_analysis).
    """

    def __init__(
        self,
        display_settings: Dict[str, Any],
        save_folder: str,
        save_file_preamble: str,
        save_figures: bool,
        interpret: bool,
        samples_as_dots_bool: bool,
        show_sample_info: str,
        box_plot_error_lim: int,
        boxplot_config: Dict[str, Any],
    ):
        """
        Initialize the QuantileBinningAnalyzer for uncertainty quantification analysis.

        Args:
            display_settings (Dict[str, Any]): Dictionary controlling which plots to generate.
                Keys include 'errors', 'error_bounds', 'jaccard', 'correlation', 'cumulative_error'.
                Each value should be boolean indicating whether to generate that plot type.
            save_folder (str): Folder path where generated plots will be saved.
            save_file_preamble (str): Prefix string for saved file names to ensure unique identification.
            save_figures (bool): If True, save plots to disk; if False, display plots interactively.
            interpret (bool): If True, execute analysis and visualization; if False, skip processing.
            samples_as_dots_bool (bool): Whether to show individual data points as dots on boxplots.
            show_sample_info (str): Mode for displaying sample size information on plots.
                Common values: "None", "text", "legend".
            box_plot_error_lim (int): Upper limit for y-axis on error boxplots (typically in mm).
            boxplot_config (Dict[str, Any]): Configuration dictionary for boxplot aesthetics including colormap, font
                sizes, figure dimensions, and styling parameters.
        """
        self.logger = logging.getLogger("qbin")
        self.display_settings = display_settings
        self.save_folder = save_folder
        self.save_file_preamble = save_file_preamble
        self.save_figures = save_figures
        self.interpret = interpret
        self.box_plot_error_lim = box_plot_error_lim
        self.boxplot_config = boxplot_config
        self.samples_as_dots_bool = samples_as_dots_bool
        self.show_sample_info = show_sample_info
        self.hatch = display_settings.get("hatch", "o")

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
                - confidence_invert: Whether to invert confidence values for analysis
                - Display and visualization settings for plot generation
                - File paths and saving configuration

        Raises:
            FileNotFoundError: If required data files are not found at specified paths.
            ValueError: If configuration parameters are invalid or incompatible.

        Example:
            >>> config = QuantileBinningConfig(
            ...     uncertainty_error_pairs=[('epistemic_uncertainty', 'localization_error', 'epistemic_uncertainty')],
            ...     models=['ResNet50', 'VGG16'],
            ...     dataset='cardiac_mri',
            ...     target_indices=[0, 1, 2, 3],  # 4 anatomical landmarks
            ...     num_bins=10,
            ...     combine_middle_bins=False,
            ...     show_individual_target_plots=True,
            ...     ind_targets_to_show=[0, 1]  # Show detailed plots for first 2 targets
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
            # Convert confidence_invert from List[List] to List[Tuple[str, bool]]
            confidence_invert_tuples = (
                [(str(item[0]), bool(item[1])) for item in config.confidence_invert] if config.confidence_invert else []
            )
            evaluate_correlations(
                eval_data["bins"],
                uncertainty_error_model_triples,
                config.num_bins,
                confidence_invert_tuples,
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
                error_scaling_factor=config.error_scaling_factor,
            )

        if self.display_settings.get("errors"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            self._plot_error_metrics(
                eval_data,
                config.models,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_per_model_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                config.target_indices,
                detailed_mode=False,
                x_label=LABEL_UNCERTAINTY_THRESHOLDED_BIN,
            )

        if self.display_settings.get("error_bounds"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            self._plot_error_bounds_metrics(
                eval_data,
                config.models,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_generic_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                False,  # detailed_mode
                config.percent_y_lim_extended,
                config.target_indices,
                x_label=LABEL_UNCERTAINTY_THRESHOLDED_BIN,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
            )

        if self.display_settings.get("jaccard"):
            uncertainty_categories = [[name, name] for name, err, unc in config.uncertainty_error_pairs]
            self._plot_jaccard_metrics(
                eval_data,
                config.models,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_generic_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                False,  # detailed_mode
                config.percent_y_lim_standard,
                config.percent_y_lim_extended,
                config.target_indices,
                x_label="Uncertainty Thresholded Bin",
                recall_y_label="Ground Truth Bins Recall",
                precision_y_label="Ground Truth Bins Precision",
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
            )

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
                - all_values_q: List of Q values to compare (e.g., [5, 10, 15, 20, 25])
                - all_fitted_save_paths: Corresponding list of data file paths for each Q value
                - combine_middle_bins: Whether to use simplified 3-bin analysis for all Q values
                - Display settings controlling which metrics to visualize
                - Individual target plotting configuration for detailed analysis

        Raises:
            FileNotFoundError: If data files for any Q value are missing or inaccessible.
            ValueError: If Q values and save paths lists have mismatched lengths.
            IndexError: If specified target indices are not present in the data.

        Example:
            >>> config = ComparingBinsConfig(
            ...     uncertainty_error_pair=('epistemic_uncertainty', 'localization_error'),
            ...     model='ResNet50',
            ...     dataset='cardiac_mri',
            ...     targets=[0, 1, 2, 3],
            ...     all_values_q=[5, 10, 15, 20],
            ...     all_fitted_save_paths=[
            ...         'data/q5_results/', 'data/q10_results/',
            ...         'data/q15_results/', 'data/q20_results/'
            ...     ],
            ...     show_individual_target_plots=True,
            ...     ind_targets_to_show=[0]  # Detailed analysis for target 0
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
        uncertainty_error_pair_list = [config.uncertainty_error_pair]

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
        for idx, num_bins in enumerate(config.all_values_q):
            eval_data = self._gather_evaluation_data(
                model_list,
                config.targets,
                config.all_fitted_save_paths[idx],
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

        num_bins_display = 3 if config.combine_middle_bins else num_bins
        category_labels = [str(q) for q in config.all_values_q]

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
            self._plot_error_metrics(
                combined_eval_data,
                model_list,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_comparing_q_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                config.targets,
                detailed_mode=True,
                x_label=LABEL_Q_NUM_BINS,
            )

        if self.display_settings.get("error_bounds"):
            uncertainty_categories = [[config.uncertainty_error_pair[0], config.uncertainty_error_pair[1]]]
            self._plot_error_bounds_metrics(
                combined_eval_data,
                model_list,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_comparing_q_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                True,  # detailed_mode
                config.percent_y_lim_extended,
                config.targets,
                x_label=LABEL_Q_NUM_BINS,
            )

        if self.display_settings.get("jaccard"):
            uncertainty_categories = [[config.uncertainty_error_pair[0], config.uncertainty_error_pair[1]]]
            self._plot_jaccard_metrics(
                combined_eval_data,
                model_list,
                uncertainty_categories,
                category_labels,
                num_bins_display,
                plot_comparing_q_boxplot,
                config.show_individual_target_plots,
                config.ind_targets_to_show or [],
                True,  # detailed_mode
                config.percent_y_lim_standard,
                config.percent_y_lim_extended,
                config.targets,
                x_label=LABEL_Q_NUM_BINS,
                recall_y_label="Ground Truth Bins Recall (%)",
                precision_y_label="Ground Truth Bins Precision (%)",
            )

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
            os.path.join(self.save_folder, "target_errors.xlsx"),
        )
        return {"errors": error_data, "jaccard": jaccard_data, "bounds": bounds_data, "bins": bins_all_targets}

    def _plot_error_metrics(
        self,
        eval_data: Dict[str, Any],
        models: List[str],
        uncertainty_categories: List[List[str]],
        category_labels: List[str],
        num_bins_display: int,
        plot_func: Callable[..., Any],
        show_individual_target_plots: bool,
        ind_targets_to_show: List[int],
        target_indices: Optional[List[int]] = None,
        detailed_mode: bool = False,
        **kwargs: Any,
    ):
        """
        Plot error-related metrics including individual errors and mean error across folds.

        This method generates three types of error plots:
        1. Main error plot aggregating all targets
        2. Individual target plots (if requested)
        3. Mean error across folds plot

        Args:
            eval_data (Dict[str, Any]): Dictionary containing evaluation data with keys:
                'errors' containing ERROR_DATA_ALL_CONCAT_NOSEP, ERROR_DATA_ALL_CONCAT_SEP, ERROR_DATA_MEAN_BINS_NOSEP.
            models (List[str]): List of model names to compare.
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations.
                Each inner list contains [uncertainty_type, error_type].
            category_labels (List[str]): Labels for x-axis categories (bins or Q values).
            num_bins_display (int): Number of bins to display on the plot.
            plot_func (Callable): Plotting function to use (plot_per_model_boxplot, plot_comparing_q_boxplot, etc.).
            show_individual_target_plots (bool): Whether to generate separate plots for individual targets.
            ind_targets_to_show (List[int]): List of target indices to plot individually. Empty list means none.
            target_indices (Optional[List[int]]): Complete list of target indices for reference. Defaults to None.
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced features. Defaults to False.
            **kwargs: Additional keyword arguments passed to the plotting function.
        """
        # Main error plot
        self._plot_metric(
            METRIC_NAME_ERROR,
            plot_func,
            eval_data["errors"][ERROR_DATA_ALL_CONCAT_NOSEP],
            models,
            uncertainty_categories,
            category_labels,
            show_sample_info=self.show_sample_info,
            show_individual_dots=self.samples_as_dots_bool,
            y_label=LABEL_LOCALIZATION_ERROR,
            to_log=True,
            convert_to_percent=False,
            num_bins_display=num_bins_display,
            y_lim_top=self.box_plot_error_lim,
            detailed_mode=detailed_mode,
            **kwargs,
        )

        # Individual target plots
        if show_individual_target_plots:
            self._plot_individual_targets(
                METRIC_NAME_ERROR,
                plot_func,
                eval_data["errors"][ERROR_DATA_ALL_CONCAT_SEP],
                ind_targets_to_show,
                uncertainty_categories,
                models,
                category_labels,
                num_bins_display,
                show_sample_info=self.show_sample_info,
                y_label=LABEL_LOCALIZATION_ERROR,
                to_log=True,
                convert_to_percent=False,
                show_individual_dots=self.samples_as_dots_bool,
                y_lim_top=self.box_plot_error_lim,
                target_indices=target_indices,
                detailed_mode=detailed_mode,
                **kwargs,
            )

        # Mean error folds plot
        self._plot_metric(
            METRIC_NAME_MEAN_ERROR_FOLDS,
            plot_func,
            eval_data["errors"][ERROR_DATA_MEAN_BINS_NOSEP],
            models,
            uncertainty_categories,
            category_labels,
            y_label=LABEL_MEAN_ERROR,
            to_log=True,
            convert_to_percent=False,
            num_bins_display=num_bins_display,
            y_lim_top=self.box_plot_error_lim,
            show_sample_info="None",
            show_individual_dots=False,
            detailed_mode=detailed_mode,
            **kwargs,
        )

    def _plot_error_bounds_metrics(
        self,
        eval_data: Dict[str, Any],
        models: List[str],
        uncertainty_categories: List[List[str]],
        category_labels: List[str],
        num_bins_display: int,
        plot_func: Callable[..., Any],
        show_individual_target_plots: bool,
        ind_targets_to_show: List[int],
        detailed_mode: bool,
        percent_y_lim_extended: int,
        target_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """
        Plot error bounds related metrics for uncertainty quantification accuracy assessment.

        This method generates error bounds plots that show how well the uncertainty estimates predict actual errors.
        It creates both aggregated and individual target plots.

        Args:
            eval_data (Dict[str, Any]): Dictionary containing evaluation data with 'bounds' key
                containing BOUNDS_DATA_ALL and BOUNDS_DATA_ALL_CONCAT_SEP data.
            models (List[str]): List of model names to analyze.
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations.
                Each inner list contains [uncertainty_type, error_type].
            category_labels (List[str]): Labels for x-axis categories (bins or Q values).
            num_bins_display (int): Number of bins to display on the plot.
            plot_func (Callable): Plotting function to use for visualization.
            show_individual_target_plots (bool): Whether to generate separate plots for individual targets.
            ind_targets_to_show (List[int]): List of target indices to plot individually.
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced features.
            percent_y_lim_extended (int): Upper limit for y-axis when plotting percentages (typically 100-120).
            target_indices (Optional[List[int]]): Complete list of target indices for reference. Defaults to None.
            **kwargs: Additional keyword arguments passed to the plotting function including
                width, y_lim_bottom, font_size_label, font_size_tick, x_label.
        """
        # Main error bounds plot
        self._plot_metric(
            METRIC_NAME_ERRORBOUND,
            plot_func,
            eval_data["bounds"][BOUNDS_DATA_ALL],
            models,
            uncertainty_categories,
            category_labels,
            y_label=LABEL_ERROR_BOUND_ACCURACY,
            to_log=False,
            convert_to_percent=True,
            num_bins_display=num_bins_display,
            show_sample_info="None",
            show_individual_dots=False,
            detailed_mode=detailed_mode,
            y_lim_top=percent_y_lim_extended,
            **kwargs,
        )

        # Individual target plots
        if show_individual_target_plots:
            self._plot_individual_targets(
                METRIC_NAME_ERRORBOUND,
                plot_func,
                eval_data["bounds"][BOUNDS_DATA_ALL_CONCAT_SEP],
                ind_targets_to_show,
                uncertainty_categories,
                models,
                category_labels,
                num_bins_display,
                show_sample_info="None",
                y_label=LABEL_ERROR_BOUND_ACCURACY,
                to_log=False,
                convert_to_percent=True,
                show_individual_dots=False,
                detailed_mode=detailed_mode,
                target_indices=target_indices,
                y_lim_top=percent_y_lim_extended,
                **kwargs,
            )

    def _plot_jaccard_metrics(
        self,
        eval_data: Dict[str, Any],
        models: List[str],
        uncertainty_categories: List[List[str]],
        category_labels: List[str],
        num_bins_display: int,
        plot_func: Callable[..., Any],
        show_individual_target_plots: bool,
        ind_targets_to_show: List[int],
        detailed_mode: bool,
        percent_y_lim_standard: int,
        percent_y_lim_extended: int,
        target_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """
        Plot Jaccard-related metrics including Jaccard index, recall, and precision.

        This method generates comprehensive Jaccard analysis plots that evaluate how well uncertainty-based bins match
        ground truth error bins. It produces four types of plots:
        1. Main Jaccard index plot (overlap between predicted and actual high-error regions)
        2. Recall plot (sensitivity - how many actual high-error cases were identified)
        3. Precision plot (specificity - how many identified cases were actually high-error)
        4. Individual target plots (if requested)

        Args:
            eval_data (Dict[str, Any]): Dictionary containing evaluation data with 'jaccard' key
                containing JACCARD_DATA_ALL, JACCARD_DATA_RECALL_ALL, JACCARD_DATA_PRECISION_ALL,
                and JACCARD_DATA_ALL_CONCAT_SEP data.
            models (List[str]): List of model names to analyze.
            uncertainty_categories (List[List[str]]): List of uncertainty-error pair combinations.
                Each inner list contains [uncertainty_type, error_type].
            category_labels (List[str]): Labels for x-axis categories (bins or Q values).
            num_bins_display (int): Number of bins to display on the plot.
            plot_func (Callable): Plotting function to use for visualization.
            show_individual_target_plots (bool): Whether to generate separate plots for individual targets.
            ind_targets_to_show (List[int]): List of target indices to plot individually.
            detailed_mode (bool): Whether to use detailed plotting mode with enhanced features.
            percent_y_lim_standard (int): Standard upper limit for y-axis when plotting percentages (typically 70).
            percent_y_lim_extended (int): Extended upper limit for y-axis for recall/precision plots (typically 120).
            target_indices (Optional[List[int]]): Complete list of target indices for reference. Defaults to None.
            **kwargs: Additional keyword arguments including recall_y_label, precision_y_label,
                width, y_lim_bottom, font_size_label, font_size_tick, x_label.
        """
        # Main Jaccard plot
        self._plot_metric(
            METRIC_NAME_JACCARD,
            plot_func,
            eval_data["jaccard"][JACCARD_DATA_ALL],
            models,
            uncertainty_categories,
            category_labels,
            y_label=LABEL_JACCARD_INDEX,
            to_log=False,
            convert_to_percent=True,
            num_bins_display=num_bins_display,
            y_lim_top=percent_y_lim_standard,
            show_sample_info="None",
            detailed_mode=detailed_mode,
            show_individual_dots=False,
            **kwargs,
        )

        # Recall plot
        recall_kwargs = kwargs.copy()
        recall_kwargs["y_label"] = kwargs.get("recall_y_label", LABEL_GROUND_TRUTH_BINS_RECALL)
        self._plot_metric(
            METRIC_NAME_RECALL_JACCARD,
            plot_func,
            eval_data["jaccard"][JACCARD_DATA_RECALL_ALL],
            models,
            uncertainty_categories,
            category_labels,
            to_log=False,
            convert_to_percent=True,
            num_bins_display=num_bins_display,
            y_lim_top=percent_y_lim_extended,
            show_sample_info="None",
            detailed_mode=detailed_mode,
            show_individual_dots=False,
            **recall_kwargs,
        )

        # Precision plot
        precision_kwargs = kwargs.copy()
        precision_kwargs["y_label"] = kwargs.get("precision_y_label", LABEL_GROUND_TRUTH_BINS_PRECISION)
        self._plot_metric(
            METRIC_NAME_PRECISION_JACCARD,
            plot_func,
            eval_data["jaccard"][JACCARD_DATA_PRECISION_ALL],
            models,
            uncertainty_categories,
            category_labels,
            to_log=False,
            convert_to_percent=True,
            num_bins_display=num_bins_display,
            y_lim_top=percent_y_lim_extended,
            show_sample_info="None",
            show_individual_dots=False,
            detailed_mode=detailed_mode,
            **precision_kwargs,
        )

        # Individual target plots for Jaccard
        if show_individual_target_plots:
            self._plot_individual_targets(
                METRIC_NAME_JACCARD,
                plot_func,
                eval_data["jaccard"][JACCARD_DATA_ALL_CONCAT_SEP],
                ind_targets_to_show,
                uncertainty_categories,
                models,
                category_labels,
                num_bins_display,
                show_sample_info="None",
                y_label=LABEL_JACCARD_INDEX,
                to_log=False,
                y_lim_top=percent_y_lim_standard,
                convert_to_percent=True,
                show_individual_dots=False,
                target_indices=target_indices,
                detailed_mode=detailed_mode,
                **kwargs,
            )

    def _plot_individual_targets(
        self,
        metric_name: str,
        plot_func: Callable[..., Any],
        sep_target_data: List[Any],
        ind_targets_to_show: List[int],
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
            plot_func (Callable): Plotting function to use (plot_per_model_boxplot, plot_comparing_q_boxplot, etc.).
            sep_target_data (List[Any]): Data separated by target. Structure varies by plotting mode:
                - For individual bin comparison: List of target data dictionaries
                - For comparing Q: List of Q-value data with target indices
            ind_targets_to_show (List[int]): List of target indices to plot individually.
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
        if plot_func == plot_comparing_q_boxplot:
            # This logic is for comparing_q mode where sep_target_data is structured differently.
            if target_indices is not None:
                for t_idx in target_indices:
                    if t_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        # Extract data for this target across all Q values
                        target_data_across_q = [q_data[t_idx] for q_data in sep_target_data]
                        individual_targets_data.append({"target_idx": t_idx, "data": target_data_across_q})
        else:
            # This logic is for individual bin comparison mode.
            for idx, target_data in enumerate(sep_target_data):
                if idx in ind_targets_to_show or ind_targets_to_show == [-1]:
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
        show_sample_info: str,
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
                plot_comparing_q_boxplot).
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
        if plot_func == plot_comparing_q_boxplot:
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

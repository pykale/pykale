# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Wenjie Zhao, mcsoft12138@outlook.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
# =============================================================================

"""
Module from the implementation of L. A. Schobs, A. J. Swift and H. Lu, "Uncertainty Estimation for Heatmap-Based Landmark Localization,"
in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

Functions related to interpreting the uncertainty quantiles from the quantile binning method in terms of:
   A) Perform Isotonic regression on uncertainty & error pairs (quantile_binning_and_est_errors)
   B) Modern configuration-based plotting functions: plot_generic_boxplot, plot_per_model_boxplot, plot_comparing_q_boxplot
   C) Cumulative error plots: plot_cumulative
   D) High-level analysis functions for QBinning: generate_fig_individual_bin_comparison, generate_fig_comparing_bins

Note: Configuration classes (BoxPlotConfig, BoxPlotData) and data processing functions are now located in kale.interpret.box_plot
for better code organization and maintainability.

"""
import logging
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter
from sklearn.isotonic import IsotonicRegression

from kale.evaluate.similarity_metrics import evaluate_correlations
from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard, get_mean_errors
from kale.interpret.box_plot import (
    BoxPlotConfig,
    BoxPlotData,
    ComparingQBoxPlotter,
    create_boxplot_config,
    create_boxplot_data,
    GenericBoxPlotter,
    PerModelBoxPlotter,
)
from kale.prepdata.tabular_transform import generate_struct_for_qbin
from kale.utils.save_xlsx import generate_summary_df


def quantile_binning_and_est_errors(
    errors: List[float],
    uncertainties: List[float],
    num_bins: int,
    type: str = "quantile",
    acceptable_thresh: float = 5,
    combine_middle_bins: bool = False,
) -> Tuple[List[List[float]], List[float]]:
    """
    Calculate quantile thresholds, and isotonically regress errors and uncertainties
    and get estimated error bounds.

    Args:
        errors (List[float]): List of errors.
        uncertainties (List[float]): List of uncertainties.
        num_bins (int): Number of quantile bins.
        type (str, optional): Type of thresholds to calculate, "quantile" recommended.
                              Defaults to "quantile".
        acceptable_thresh (float, optional): Acceptable error threshold. Only relevant
                                             if type="error-wise". Defaults to 5.
        combine_middle_bins (bool, optional): Whether to combine middle bins.
                                              Defaults to False.

    Returns:
        Tuple[List[List[float]], List[float]]: List of quantile thresholds and
                                               estimated error bounds.
    """

    if len(errors) != len(uncertainties):
        raise ValueError(
            "Length of errors and uncertainties must be the same. errors is length %s and uncertainties is length %s"
            % (len(errors), len(uncertainties))
        )

    valid_types = {"quantile", "error-wise"}
    if type not in valid_types:
        raise ValueError("results: type must be one of %r. " % valid_types)

    # Isotonically regress line
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)

    _ = ir.fit_transform(uncertainties, errors)

    uncert_boundaries = []
    estimated_errors = []

    # Estimate error bounds for each quantile bin
    if type == "quantile":
        quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)[: num_bins - 1]
        for q in range(len(quantiles)):
            q_conf_higher = [np.quantile(uncertainties, quantiles[q])]
            q_error_higher = ir.predict(q_conf_higher)

            estimated_errors.append(q_error_higher[0])
            uncert_boundaries.append(q_conf_higher)

    elif type == "error_wise":
        quantiles = np.arange(num_bins - 1)
        estimated_errors = [[(acceptable_thresh * x)] for x in quantiles]

        uncert_boundaries = [(ir.predict(x)).tolist() for x in estimated_errors]
        raise NotImplementedError("error_wise Quantile Binning not implemented yet")

    # IF combine bins, we grab only the values for the two outer bins
    if combine_middle_bins:
        estimated_errors = [estimated_errors[0], estimated_errors[-1]]
        uncert_boundaries = [uncert_boundaries[0], uncert_boundaries[-1]]

    return uncert_boundaries, estimated_errors


def plot_generic_boxplot(data: BoxPlotData, config: BoxPlotConfig) -> None:
    """
    Create generic multi-model boxplot for uncertainty quantification analysis.

    This function generates boxplots comparing multiple models across different uncertainty
    bins or categories. It's designed for comparative analysis of model performance in
    medical imaging uncertainty quantification workflows.

    Args:
        data (BoxPlotData): Data container object containing all required inputs:
            - colormap: Matplotlib colormap name for consistent visual distinction across plots.
            - evaluation_data_by_bins: Dictionary mapping model+uncertainty combinations to binned data
            - uncertainty_categories: List of uncertainty type groupings
            - models: List of model identifiers for comparison
            - category_labels: Labels for x-axis categories (bins, quantiles, etc.)
            - num_bins: Number of bins for data organization
        config (BoxPlotConfig): Configuration object containing all display parameters:
            - x_label: Label for x-axis
            - y_label: Label for y-axis
            - save_path: File path for saving the figure (optional)
            - show_individual_dots: Whether to show individual data points
            - y_lim_top/y_lim_bottom: Y-axis limits
            - font_size_label/font_size_tick: Font size settings
            - Additional styling and display options

    Returns:
        None: The function displays and/or saves the figure based on configuration.

    Example:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=error_data,
        ...     uncertainty_categories=[['epistemic'], ['aleatoric']],
        ...     models=['ResNet50', 'VGG16']
        ... )
        >>> config = create_boxplot_config(
        ...     colormap='Set1',
        ...     x_label="Uncertainty Bins",
        ...     y_label="Localization Error (mm)",
        ...     save_path="comparison.pdf"
        ... )
        >>> plot_generic_boxplot(data, config)
    """
    plotter = GenericBoxPlotter(data, config)
    plotter.process_data()
    plotter.draw_boxplot()


def plot_per_model_boxplot(data: BoxPlotData, config: BoxPlotConfig) -> None:
    """
    Generate per-model boxplot for individual model performance analysis.

    This function creates boxplots that focus on analyzing the performance of individual models
    across different uncertainty bins. It's particularly useful for detailed model-specific
    uncertainty quantification analysis in medical imaging applications.

    Args:
        data (BoxPlotData): Data container object containing all required inputs:
            - colormap: Matplotlib colormap name for consistent visual distinction across plots.
            - evaluation_data_by_bins: Dictionary mapping model+uncertainty combinations to binned data
            - uncertainty_categories: List of uncertainty type groupings for the specific model
            - models: List containing the model identifier(s) being analyzed
            - category_labels: Labels for x-axis categories (uncertainty bins, thresholds, etc.)
            - num_bins: Number of bins used for uncertainty organization
        config (BoxPlotConfig): Configuration object containing all display parameters:
            - x_label: Label for x-axis (e.g., "Uncertainty Thresholded Bin")
            - y_label: Label for y-axis (e.g., "Localization Error (mm)")
            - save_path: File path for saving the figure (optional)
            - show_individual_dots: Whether to display individual data points as dots
            - show_sample_info: Mode for displaying sample size information
            - y_lim_top/y_lim_bottom: Y-axis limits for consistent scaling
            - to_log: Whether to use logarithmic y-axis scaling
            - Additional styling and formatting options

    Returns:
        None: The function displays and/or saves the figure based on configuration settings.

    Example:
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=model_error_data,
        ...     uncertainty_categories=[['epistemic', 'aleatoric']],
        ...     models=['ResNet50'],
        ...     category_labels=['Low', 'Medium', 'High']
        ... )
        >>> config = create_boxplot_config(
        ...     colormap='Set1',
        ...     x_label="Uncertainty Thresholded Bin",
        ...     y_label="Localization Error (mm)",
        ...     to_log=True,
        ...     save_path="resnet_analysis.pdf"
        ... )
        >>> plot_per_model_boxplot(data, config)
    """
    plotter = PerModelBoxPlotter(data, config)
    plotter.process_data()
    plotter.draw_boxplot()


def plot_comparing_q_boxplot(data: BoxPlotData, config: BoxPlotConfig) -> None:
    """
    Create boxplot comparing different Q values for quantile threshold optimization.

    This function generates boxplots that compare the impact of different quantile thresholds
    (Q values) on uncertainty quantification performance. It's essential for optimizing
    binning strategies and understanding threshold sensitivity in medical imaging applications.

    Args:
        data (BoxPlotData): Data container object containing all required inputs:
            - colormap: Matplotlib colormap name for visual distinction
            - evaluation_data_by_bins: List of dictionaries, one per Q value, containing binned evaluation data
            - uncertainty_categories: List of uncertainty type groupings (typically single uncertainty type)
            - models: List containing the model identifier(s) being compared across Q values
            - category_labels: Labels for x-axis categories representing Q values (e.g., ['Q=5', 'Q=10', 'Q=15'])
            - num_bins: Number of bins used within each Q value configuration
        config (BoxPlotConfig): Configuration object containing all display parameters:
            - x_label: Label for x-axis (e.g., "Q (# Bins)")
            - y_label: Label for y-axis (e.g., "Localization Error (mm)")
            - hatch_type: Hatching pattern for distinguishing Q-comparison plots (e.g., '///')
            - color: Base color for the boxplots
            - save_path: File path for saving the figure (optional)
            - show_individual_dots: Whether to display individual data points
            - y_lim_top/y_lim_bottom: Y-axis limits
            - to_log: Whether to use logarithmic y-axis scaling
            - Additional styling and formatting options

    Returns:
        None: The function displays and/or saves the figure based on configuration settings.

    Raises:
        ValueError: If required fields (evaluation_data_by_bins, uncertainty_categories, models) are missing.

    Example:
        >>> # Compare Q=5 vs Q=10 vs Q=15 quantile thresholds
        >>> data = create_boxplot_data(
        ...     evaluation_data_by_bins=[q5_results, q10_results, q15_results],
        ...     uncertainty_categories=[['epistemic']],
        ...     models=['ResNet50'],
        ...     category_labels=['Q=5', 'Q=10', 'Q=15']
        ... )
        >>> config = create_boxplot_config(
        ...     x_label="Q (# Bins)",
        ...     y_label="Localization Error (mm)",
        ...     hatch_type='///',
        ...     colormap='Set1',
        ...     save_path="q_comparison.pdf"
        ... )
        >>> plot_comparing_q_boxplot(data, config)
    """
    if not all([data.evaluation_data_by_bins, data.uncertainty_categories, data.models]):
        raise ValueError(
            "For comparing_q plots, data must include evaluation_data_by_bins, uncertainty_categories, and models"
        )

    plotter = ComparingQBoxPlotter(data, config)
    plotter.draw_boxplot()


def plot_cumulative(
    colormap: str,
    data_struct: Dict[str, pd.DataFrame],
    models: List[str],
    uncertainty_types: List[Tuple[str, str]],
    bins: Union[List[int], np.ndarray],
    title: str,
    compare_to_all: bool = False,
    save_path: Optional[str] = None,
    error_scaling_factor: float = 1,
) -> None:
    """
    Plots cumulative errors.

    Args:
        colormap (str): Matplotlib colormap name for consistent visual distinction across plots.
        data_struct: A dictionary containing the dataframes for each model.
        models: A list of models we want to compare, keys in `data_struct`.
        uncertainty_types: A list of lists describing the different uncertainty combinations to test.
        bins: A list of bins to show error form.
        title: The title of the plot.
        compare_to_all: Whether to compare the given subset of bins to all the data (default=False).
        save_path: The path to save plot to. If None, displays on screen (default=None).
        error_scaling_factor (float, optional): Scaling factor for error. Defaults to 1.0.
    """

    # make sure bins is a list and not a single value
    bins = [bins] if not isinstance(bins, (list, np.ndarray)) else bins

    plt.style.use("ggplot")

    _ = plt.figure()

    ax = plt.gca()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax.set_xlabel("Error (mm)", fontsize=10)
    ax.set_ylabel("Number of images in %", fontsize=10)
    plt.title(title)

    ax.set_xscale("log")
    line_styles = [":", "-", "dotted", "-."]
    colors = colormaps.get_cmap(colormap)(np.arange(len(uncertainty_types) + 1))
    for i, (uncert_pair) in enumerate(uncertainty_types):
        uncertainty = (uncert_pair)[0]
        color = colors[i]
        for hash_idx, model_type in enumerate(models):
            line = line_styles[hash_idx]

            # Filter only the bins selected
            dataframe = data_struct[model_type]
            model_un_errors = (
                dataframe[dataframe[uncertainty + " Uncertainty bins"].isin(bins)][uncertainty + " Error"].values
                * error_scaling_factor
            )

            p = 100 * np.arange(len(model_un_errors)) / (len(model_un_errors) - 1)

            sorted_errors = np.sort(model_un_errors)

            ax.plot(
                sorted_errors,
                p,
                label=model_type + " " + uncertainty,
                color=color,
                linestyle=line,
                dash_capstyle="round",
            )

            if compare_to_all:
                dataframe = data_struct[model_type]
                model_un_errors = dataframe[uncertainty + " Error"].values * error_scaling_factor

                p = 100 * np.arange(len(model_un_errors)) / (len(model_un_errors) - 1)

                sorted_errors = np.sort(model_un_errors)
                line = line_styles[len(models) + hash_idx]
                ax.plot(
                    sorted_errors,
                    p,
                    label=model_type + " " + uncertainty,
                    color=color,
                    linestyle=line,
                    dash_capstyle="round",
                )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop={"size": 10})
    plt.axvline(x=5, color=colors[len(uncertainty_types)])

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    plt.xticks([1, 2, 3, 4, 5, 10, 20, 30])

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    if save_path is not None:
        plt.savefig(save_path + "cumulative_error.pdf", dpi=100, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.show()
        plt.close()


def generate_fig_individual_bin_comparison(data: Tuple, display_settings: dict) -> None:
    """
    [Legacy interface] Wrapper for calling QuantileBinningAnalyzer.run_individual_bin_comparison.

    Note: This is a legacy interface kept for backward compatibility.
    In new code, it's recommended to directly instantiate and use the QuantileBinningAnalyzer class.
    """

    # 1. Unpack configuration parameters from data tuple
    (
        _,
        _,
        _,
        _,
        _,
        colormap,
        save_folder,
        save_file_preamble,
        _,
        save_figures_bool,
        _,
        samples_as_dots_bool,
        show_sample_info,
        box_plot_error_lim,
        _,
        interpret,
        _,
        _,
        _,
    ) = data

    # 2. Prepare configuration for boxplot aesthetics
    boxplot_aesthetic_config = {"colormap": colormap, "hatch_type": display_settings.get("hatch", "")}

    # 3. Instantiate and run the new analyzer
    analyzer = QuantileBinningAnalyzer(
        display_settings=display_settings,
        save_folder=save_folder,
        save_file_preamble=save_file_preamble,
        save_figures=save_figures_bool,
        interpret=interpret,
        samples_as_dots_bool=samples_as_dots_bool,
        show_sample_info=show_sample_info,
        box_plot_error_lim=box_plot_error_lim,
        boxplot_config=boxplot_aesthetic_config,
    )
    analyzer.run_individual_bin_comparison(data)


def generate_fig_comparing_bins(data: Tuple, display_settings: Dict[str, Any]) -> None:
    """
    [Legacy interface] Wrapper for calling QuantileBinningAnalyzer.run_comparing_bins_analysis.

    Note: This is a legacy interface kept for backward compatibility.
    In new code, it's recommended to directly instantiate and use the QuantileBinningAnalyzer class.
    """

    # 1. Unpack configuration parameters from data tuple
    (
        _,
        _,
        _,
        _,
        _,
        colormap,
        _,
        save_folder,
        save_file_preamble,
        _,
        save_figures_bool,
        samples_as_dots_bool,
        show_sample_info,
        box_plot_error_lim,
        _,
        interpret,
        _,
        _,
        _,
    ) = data

    # 2. Prepare configuration for boxplot aesthetics
    boxplot_aesthetic_config = {"colormap": colormap, "hatch_type": display_settings.get("hatch", "")}

    # 3. Instantiate and run the new analyzer
    analyzer = QuantileBinningAnalyzer(
        display_settings=display_settings,
        save_folder=save_folder,
        save_file_preamble=save_file_preamble,
        save_figures=save_figures_bool,
        interpret=interpret,
        samples_as_dots_bool=samples_as_dots_bool,
        show_sample_info=show_sample_info,
        box_plot_error_lim=box_plot_error_lim,
        boxplot_config=boxplot_aesthetic_config,
    )
    analyzer.run_comparing_bins_analysis(data)


class QuantileBinningAnalyzer:
    """
    A class for performing and visualizing Quantile Binning uncertainty analysis.

    This class encapsulates the functionality of the original generate_fig_individual_bin_comparison
    and generate_fig_comparing_bins functions to reduce code duplication and improve maintainability.
    It provides two core analysis modes:
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
        Initialize the analyzer.

        Args:
            display_settings (Dict[str, Any]): Dictionary controlling which plots to generate (e.g., {'errors': True, 'jaccard': True}).
            save_folder (str): Folder path to save generated plots.
            save_file_preamble (str): Prefix for saved file names.
            save_figures (bool): If True, save plots; otherwise, display plots.
            interpret (bool): If True, execute analysis and visualization.
            boxplot_config (Dict[str, Any]): Generic aesthetic configuration for boxplots.
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
        self.hatch = display_settings.get("hatch", "")

    def _get_save_location(self, suffix: str) -> Optional[str]:
        """Construct and return the save path for charts, or return None if not saving."""
        if not self.save_figures:
            return None

        filename_suffix = "_dotted" if self.samples_as_dots_bool else "_undotted"
        return os.path.join(self.save_folder, f"{self.save_file_preamble}{filename_suffix}_{suffix}.pdf")

    def _gather_evaluation_data(
        self,
        models: List[str],
        targets: List[int],
        save_path_pre: str,
        dataset: str,
        num_bins: int,
        uncertainty_pairs: List[Tuple],
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
            uncertainty_pairs (List[Tuple]): Uncertainty-error pair configurations.
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
            [["all mean error bins nosep", "All Targets"]],
            "Mean error",
            os.path.join(self.save_folder, "target_errors.xlsx"),
        )
        return {"errors": error_data, "jaccard": jaccard_data, "bounds": bounds_data, "bins": bins_all_targets}

    def _plot_individual_targets(
        self,
        metric_name: str,
        plot_func: Callable,
        sep_target_data: List[Any],
        ind_targets_to_show: List[int],
        uncertainty_categories: List[List[str]],
        models: List[str],
        category_labels: List[str],
        num_bins_display: int,
        target_indices: Optional[List[int]],
        **kwargs: Any,
    ):
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
            target_data = target_info["data"]
            self.logger.info(f"Plotting individual {metric_name} for T{target_idx}")

            boxplot_data_ind = create_boxplot_data(
                evaluation_data_by_bins=target_data,
                uncertainty_categories=uncertainty_categories,
                models=models,
                category_labels=category_labels,
                num_bins=num_bins_display,
            )
            boxplot_config_ind = create_boxplot_config(
                save_path=self._get_save_location(f"{metric_name}_target_{target_idx}"),
                **self.boxplot_config,
                **kwargs,
            )
            plot_func(boxplot_data_ind, boxplot_config_ind)

    def _plot_metric(
        self,
        metric_name: str,
        plot_func: Callable,
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
        Universal metric plotting function for drawing boxplots of errors, Jaccard index, or error bounds.

        Args:
            metric_name (str): The metric to be plotted ('errors', 'jaccard', 'bounds').
            plot_mode (str): The plotting mode ('individual' or 'comparing').
            data_by_q (List[Dict]): Raw data list organized by Q values.
            models (List[str]): List of model names.
            uncertainty_categories (List[Tuple]): List of uncertainty types.
            category_labels (List[str]): X-axis labels.
            num_bins_display (int): Number of bins to display.
            show_individual_targets (bool): Whether to plot for each target separately.
            ind_targets_to_show (List[int]): List of target indices to show individually.
        """

        # Plot aggregated charts for all targets
        self.logger.info(f"Plotting {metric_name} for all targets.")

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

        boxplot_data = create_boxplot_data(
            evaluation_data_by_bins=plot_data_all,
            uncertainty_categories=uncertainty_categories,
            models=models,
            category_labels=category_labels,
            num_bins=num_bins_display,
        )

        boxplot_config = create_boxplot_config(
            x_label=x_label,
            y_label=y_label,
            to_log=to_log,
            save_path=self._get_save_location(f"{metric_name}_all_targets"),
            convert_to_percent=convert_to_percent,
            show_sample_info=show_sample_info,
            show_individual_dots=show_individual_dots,
            detailed_mode=detailed_mode,
            **kwargs,
            **self.boxplot_config,
        )
        plot_func(boxplot_data, boxplot_config)

    def run_individual_bin_comparison(self, data: Tuple) -> None:
        """
        Execute comparative analysis of different models and uncertainty types at fixed bin counts.
        """
        if not self.interpret:
            return

        (
            uncertainty_error_pairs,
            models_to_compare,
            dataset,
            target_indices,
            num_bins,
            _,
            _,
            _,
            combine_middle_bins,
            _,
            confidence_invert,
            _,
            _,
            _,
            show_individual_target_plots,
            _,
            num_folds,
            ind_targets_to_show,
            error_scaling_factor,
        ) = data

        # Data loading and computation
        eval_data = self._gather_evaluation_data(
            models_to_compare,
            target_indices,
            self.save_folder,
            dataset,
            num_bins,
            uncertainty_error_pairs,
            num_folds,
            error_scaling_factor,
            combine_middle_bins,
        )

        num_bins_display = 3 if combine_middle_bins else num_bins
        category_labels = [rf"$B_{{{num_bins_display + 1 - (i + 1)}}}$" for i in range(num_bins_display + 1)]
        # --- Start plotting ---
        if self.display_settings.get("correlation"):
            colormap = self.boxplot_config.get("colormap", "Set1")
            evaluate_correlations(
                eval_data["bins"],
                uncertainty_error_pairs,
                num_bins,
                confidence_invert,
                num_folds=num_folds,
                colormap=colormap,
                error_scaling_factor=error_scaling_factor,
                combine_middle_bins=combine_middle_bins,
                save_path=self.save_folder if self.save_figures else None,
                to_log=True,
            )

        if self.display_settings.get("cumulative_error"):
            colormap = self.boxplot_config.get("colormap", "Set1")
            plot_cumulative(
                colormap,
                eval_data["bins"],
                models_to_compare,
                uncertainty_error_pairs,
                np.arange(num_bins),
                f"Cumulative error for ALL predictions, dataset {dataset}",
                save_path=self.save_folder if self.save_figures else None,
                error_scaling_factor=error_scaling_factor,
            )

        if self.display_settings.get("errors"):
            self._plot_metric(
                "error",
                plot_per_model_boxplot,
                eval_data["errors"]["all error concat bins targets nosep"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                show_sample_info=self.show_sample_info,
                show_individual_dots=self.samples_as_dots_bool,
                x_label="Uncertainty Thresholded Bin",
                y_label="Localization Error (mm)",
                to_log=True,
                convert_to_percent=False,
                num_bins_display=num_bins_display,
                y_lim_top=self.box_plot_error_lim,
                detailed_mode=False,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "error",
                    plot_per_model_boxplot,
                    eval_data["errors"]["all error concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pairs,
                    models_to_compare,
                    category_labels,
                    show_sample_info=self.show_sample_info,
                    show_individual_dots=self.samples_as_dots_bool,
                    num_bins_display=num_bins_display,
                    x_label="Uncertainty Thresholded Bin",
                    y_label="Localization Error (mm)",
                    to_log=True,
                    convert_to_percent=False,
                    y_lim_top=120,
                    detailed_mode=False,
                    target_indices=target_indices,
                )

            self._plot_metric(
                "mean_error_folds",
                plot_per_model_boxplot,
                eval_data["errors"]["all mean error bins nosep"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Mean Error (mm)",
                to_log=True,
                convert_to_percent=False,
                num_bins_display=num_bins_display,
                y_lim_top=self.box_plot_error_lim,
                detailed_mode=True,
                show_sample_info="None",
                show_individual_dots=False,
            )

        if self.display_settings.get("error_bounds"):
            self._plot_metric(
                "errorbound",
                plot_generic_boxplot,
                eval_data["bounds"]["Error Bounds All"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Error Bound Accuracy (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                detailed_mode=False,
                show_sample_info="None",
                show_individual_dots=False,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "errorbound",
                    plot_generic_boxplot,
                    eval_data["bounds"]["all errorbound concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pairs,
                    models_to_compare,
                    category_labels,
                    show_sample_info="None",
                    num_bins_display=num_bins_display,
                    y_lim_top=120,
                    width=0.2,
                    to_log=False,
                    y_lim_bottom=-2,
                    font_size_label=30,
                    font_size_tick=30,
                    show_individual_dots=False,
                    x_label="Uncertainty Thresholded Bin",
                    y_label="Error Bound Accuracy (%)",
                    detailed_mode=True,
                    convert_to_percent=True,
                    target_indices=target_indices,
                )

        if self.display_settings.get("jaccard"):
            self._plot_metric(
                "jaccard",
                plot_generic_boxplot,
                eval_data["jaccard"]["Jaccard All"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Jaccard Index (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                y_lim_top=70,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                detailed_mode=False,
                show_sample_info="None",
                show_individual_dots=False,
            )

            self._plot_metric(
                "recall_jaccard",
                plot_generic_boxplot,
                eval_data["jaccard"]["Recall All"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Recall",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                detailed_mode=False,
                show_sample_info="None",
                show_individual_dots=False,
            )

            self._plot_metric(
                "precision_jaccard",
                plot_generic_boxplot,
                eval_data["jaccard"]["Precision All"],
                models_to_compare,
                uncertainty_error_pairs,
                category_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Precision",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                detailed_mode=False,
                show_sample_info="None",
                show_individual_dots=False,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "jaccard",
                    plot_generic_boxplot,
                    eval_data["jaccard"]["all jacc concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pairs,
                    models_to_compare,
                    category_labels,
                    show_sample_info="None",
                    num_bins_display=num_bins_display,
                    y_lim_top=70,
                    width=0.2,
                    y_lim_bottom=-2,
                    font_size_label=30,
                    font_size_tick=30,
                    show_individual_dots=False,
                    detailed_mode=False,
                    x_label="Uncertainty Thresholded Bin",
                    y_label="Jaccard Index (%)",
                    to_log=False,
                    convert_to_percent=True,
                    target_indices=target_indices,
                )

    def run_comparing_bins_analysis(self, data: Tuple) -> None:
        """
        Execute comparative analysis of different bin counts (Q values).
        """
        if not self.interpret:
            return

        (
            uncertainty_error_pair,
            model,
            dataset,
            targets,
            all_values_q,
            _,
            all_fitted_save_paths,
            _,
            _,
            combine_middle_bins,
            _,
            _,
            _,
            _,
            show_individual_target_plots,
            _,
            num_folds,
            ind_targets_to_show,
            error_scaling_factor,
        ) = data

        model_list = [model]
        uncertainty_error_pair_list = [uncertainty_error_pair]

        # Collect data for each Q value
        all_eval_data_by_q: Dict[str, List[Any]] = {
            "all error concat bins targets nosep": [],
            "all error concat bins targets sep all": [],
            "all mean error bins nosep": [],
            "Error Bounds All": [],
            "all errorbound concat bins targets sep all": [],
            "Jaccard All": [],
            "Recall All": [],
            "Precision All": [],
            "all jacc concat bins targets sep all": [],
        }
        for idx, num_bins in enumerate(all_values_q):
            eval_data = self._gather_evaluation_data(
                model_list,
                targets,
                all_fitted_save_paths[idx],
                dataset,
                num_bins,
                uncertainty_error_pair_list,
                num_folds,
                error_scaling_factor,
                combine_middle_bins,
            )
            all_eval_data_by_q["all error concat bins targets nosep"].append(
                eval_data["errors"]["all error concat bins targets nosep"]
            )
            all_eval_data_by_q["all error concat bins targets sep all"].append(
                eval_data["errors"]["all error concat bins targets sep all"]
            )
            all_eval_data_by_q["all mean error bins nosep"].append(eval_data["errors"]["all mean error bins nosep"])
            all_eval_data_by_q["Error Bounds All"].append(eval_data["bounds"]["Error Bounds All"])
            all_eval_data_by_q["all errorbound concat bins targets sep all"].append(
                eval_data["bounds"]["all errorbound concat bins targets sep all"]
            )
            all_eval_data_by_q["Jaccard All"].append(eval_data["jaccard"]["Jaccard All"])
            all_eval_data_by_q["Recall All"].append(eval_data["jaccard"]["Recall All"])
            all_eval_data_by_q["Precision All"].append(eval_data["jaccard"]["Precision All"])
            all_eval_data_by_q["all jacc concat bins targets sep all"].append(
                eval_data["jaccard"]["all jacc concat bins targets sep all"]
            )

        num_bins_display = 3 if combine_middle_bins else num_bins
        category_labels = [str(q) for q in all_values_q]

        # --- Start plotting ---
        if self.display_settings.get("errors"):
            self._plot_metric(
                "error",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["all error concat bins targets nosep"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info=self.show_sample_info,
                x_label="Q (# Bins)",
                y_label="Localization Error (mm)",
                to_log=True,
                y_lim_top=self.box_plot_error_lim,
                convert_to_percent=False,
                num_bins_display=num_bins_display,
                show_individual_dots=self.samples_as_dots_bool,
                detailed_mode=True,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "error",
                    plot_comparing_q_boxplot,
                    all_eval_data_by_q["all error concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pair_list,
                    model_list,
                    category_labels,
                    num_bins_display,
                    show_sample_info=self.show_sample_info,
                    x_label="Q (# Bins)",
                    y_label="Localization Error (mm)",
                    to_log=True,
                    y_lim_top=self.box_plot_error_lim,
                    convert_to_percent=False,
                    show_individual_dots=self.samples_as_dots_bool,
                    detailed_mode=True,
                    target_indices=targets,
                )

            self._plot_metric(
                "mean_error_folds",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["all mean error bins nosep"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info="None",
                y_lim_top=self.box_plot_error_lim,
                x_label="Q (# Bins)",
                y_label="Mean Error (mm)",
                to_log=True,
                convert_to_percent=False,
                num_bins_display=num_bins_display,
                show_individual_dots=False,
                detailed_mode=True,
            )

        if self.display_settings.get("error_bounds"):
            self._plot_metric(
                "errorbound",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["Error Bounds All"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info="None",
                y_lim_top=100,
                x_label="Q (# Bins)",
                y_label="Error Bound Accuracy (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                show_individual_dots=False,
                detailed_mode=True,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "errorbound",
                    plot_comparing_q_boxplot,
                    all_eval_data_by_q["all errorbound concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pair_list,
                    model_list,
                    category_labels,
                    num_bins_display,
                    show_sample_info="None",
                    x_label="Q (# Bins)",
                    y_label="Error Bound Accuracy (%)",
                    to_log=False,
                    y_lim_top=100,
                    convert_to_percent=True,
                    show_individual_dots=False,
                    detailed_mode=True,
                    target_indices=targets,
                )

        if self.display_settings.get("jaccard"):
            self._plot_metric(
                "jaccard",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["Jaccard All"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info="None",
                y_lim_top=70,
                x_label="Q (# Bins)",
                y_label="Jaccard Index (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                show_individual_dots=False,
                detailed_mode=True,
            )

            self._plot_metric(
                "recall_jaccard",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["Recall All"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info="None",
                y_lim_top=120,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bins Recall (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                show_individual_dots=False,
                detailed_mode=True,
            )

            self._plot_metric(
                "precision_jaccard",
                plot_comparing_q_boxplot,
                all_eval_data_by_q["Precision All"],
                model_list,
                uncertainty_error_pair_list,
                category_labels,
                show_sample_info="None",
                y_lim_top=120,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bins Precision (%)",
                to_log=False,
                convert_to_percent=True,
                num_bins_display=num_bins_display,
                show_individual_dots=False,
                detailed_mode=True,
            )

            if show_individual_target_plots:
                self._plot_individual_targets(
                    "jaccard",
                    plot_comparing_q_boxplot,
                    all_eval_data_by_q["all jacc concat bins targets sep all"],
                    ind_targets_to_show,
                    uncertainty_error_pair_list,
                    model_list,
                    category_labels,
                    num_bins_display,
                    show_sample_info="None",
                    x_label="Q (# Bins)",
                    y_label="Jaccard Index (%)",
                    to_log=False,
                    y_lim_top=70,
                    convert_to_percent=True,
                    show_individual_dots=False,
                    detailed_mode=True,
                    target_indices=targets,
                )

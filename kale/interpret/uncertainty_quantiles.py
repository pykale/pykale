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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    Generate comprehensive figures comparing localization errors, error bounds accuracy, and Jaccard index across uncertainty bins.

    This function creates a complete set of analysis figures for individual bin comparison in uncertainty
    quantification workflows. It generates boxplots for error analysis, bound accuracy evaluation, and
    Jaccard index assessment across different uncertainty bins for medical imaging applications.

    Args:
        data (Tuple): A tuple containing various inputs needed to generate the figures. The tuple should include the following elements:
            - uncertainty_error_pairs (List[Tuple[int, float]]): List of tuples specifying uncertainty thresholds
              and corresponding error thresholds for binning the data.
            - models_to_compare (List[str]): List of model names to include in the comparative analysis.
            - dataset (str): Name of the dataset being analyzed for labeling and file naming.
            - target_indices (List[int]): List of anatomical target indices to include in the analysis.
            - num_bins (int): Number of uncertainty bins to use for quantile-based thresholding.
            - colormap (str, optional): Matplotlib colormap name for consistent visual distinction across plots. Defaults to "Set1".
            - save_folder (str): Directory path where all generated figures will be saved.
            - save_file_preamble (str): String prefix for all generated figure filenames.
            - combine_middle_bins (bool): Whether to combine middle bins into a single category.
            - save_figures_bool (bool): Whether to save generated figures to disk or display them.
            - confidence_invert (bool): Whether to invert confidence values to uncertainty values.
            - samples_as_dots_bool (bool): Whether to overlay individual data points on boxplots.
            - show_sample_info_mode (str): Mode for displaying sample size information on plots.
            - box_plot_error_lim (float): Upper y-axis limit for error boxplots.
            - show_individual_target_plots (bool): Whether to generate separate plots for each target.
            - interpret (bool): Whether to perform interpretation analysis and visualization.
            - num_folds (int): Number of cross-validation folds used in the analysis.
            - ind_targets_to_show (List[int]): List of specific target indices for individual plots.
            - error_scaling_factor (float, optional): Multiplicative factor for error value scaling. Defaults to 1.0.

        display_settings (dict): Dictionary containing boolean flags for controlling figure generation:
            - 'correlation': Whether to generate correlation analysis plots.
            - 'cumulative_error': Whether to generate cumulative error distribution plots.
            - 'errors': Whether to generate error analysis boxplots.
            - 'error_bounds': Whether to generate error bound accuracy plots.
            - 'jaccard': Whether to generate Jaccard index analysis plots.
            - 'hatch': String representing the type of hatch pattern to use in the plots.
    Returns:
        None: The function generates and saves/displays figures based on the configuration.

    Example:
        >>> data_tuple = (
        ...     [('epistemic', 0.1)], ['ResNet50'], 'cardiac_mri', [0, 1, 2],
        ...     5, ['red', 'blue'], '/results/', 'analysis_', False, True,
        ...     False, True, 'count', 10.0, True, True, 3, [-1], 1.0
        ... )
        >>> settings = {
        ...     'correlation': True, 'cumulative_error': True,
        ...     'errors': True, 'error_bounds': True, 'jaccard': True
        ... }
        >>> generate_fig_individual_bin_comparison(data_tuple, settings)
    """
    logger = logging.getLogger("qbin")
    [
        uncertainty_error_pairs,
        models_to_compare,
        dataset,
        target_indices,
        num_bins,
        colormap,
        save_folder,
        save_file_preamble,
        combine_middle_bins,
        save_figures_bool,
        confidence_invert,
        samples_as_dots_bool,
        show_sample_info_mode,
        box_plot_error_lim,
        show_individual_target_plots,
        interpret,
        num_folds,
        ind_targets_to_show,
        error_scaling_factor,
    ] = data

    # If combining the middle bins we just have the 2 edge bins, and the combined middle ones.

    bins_all_targets, bins_targets_sep, bounds_all_targets, bounds_targets_sep = generate_struct_for_qbin(
        models_to_compare, target_indices, save_folder, dataset
    )
    hatch = display_settings.get("hatch", "")
    # Get mean errors bin-wise, get all errors concatenated together bin-wise, and seperate by target.
    all_error_data_dict = get_mean_errors(
        bins_all_targets,
        uncertainty_error_pairs,
        num_bins,
        target_indices,
        num_folds=num_folds,
        error_scaling_factor=error_scaling_factor,
        combine_middle_bins=combine_middle_bins,
    )
    all_error_data = all_error_data_dict["all mean error bins nosep"]

    all_bins_concat_targets_nosep_error = all_error_data_dict[
        "all error concat bins targets nosep"
    ]  # shape is [num bins]

    all_bins_concat_targets_sep_all_error = all_error_data_dict[
        "all error concat bins targets sep all"
    ]  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

    # Get jaccard
    all_jaccard_data_dict = evaluate_jaccard(
        bins_all_targets,
        uncertainty_error_pairs,
        num_bins,
        target_indices,
        num_folds=num_folds,
        combine_middle_bins=combine_middle_bins,
    )
    all_jaccard_data = all_jaccard_data_dict["Jaccard All"]
    all_recall_data = all_jaccard_data_dict["Recall All"]
    all_precision_data = all_jaccard_data_dict["Precision All"]

    all_bins_concat_targets_sep_all_jacc = all_jaccard_data_dict[
        "all jacc concat bins targets sep all"
    ]  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

    bound_return_dict = evaluate_bounds(
        bounds_all_targets,
        bins_all_targets,
        uncertainty_error_pairs,
        num_bins,
        target_indices,
        num_folds,
        combine_middle_bins=combine_middle_bins,
    )

    all_bound_data = bound_return_dict["Error Bounds All"]

    all_bins_concat_targets_sep_all_errorbound = bound_return_dict[
        "all errorbound concat bins targets sep all"
    ]  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

    generate_summary_df(
        all_error_data_dict,
        [["all mean error bins nosep", "All Targets"]],
        "Mean error",
        os.path.join(save_folder, "target_errors.xlsx"),
    )

    if interpret:
        # If we have combined the middle bins, we are only displaying 3 bins (outer edges, and combined middle bins).
        if combine_middle_bins:
            num_bins_display = 3
        else:
            num_bins_display = num_bins

        if save_figures_bool:
            save_location = save_folder
        else:
            save_location = None

        # Plot piecewise linear regression for error/uncertainty prediction.
        if display_settings["correlation"]:
            _ = evaluate_correlations(
                bins_all_targets,
                uncertainty_error_pairs,
                num_bins,
                confidence_invert,
                num_folds=num_folds,
                colormap=colormap,
                error_scaling_factor=error_scaling_factor,
                combine_middle_bins=combine_middle_bins,
                save_path=save_location,
                to_log=True,
            )

        # Plot cumulative error figure for all predictions
        if display_settings["cumulative_error"]:
            plot_cumulative(
                colormap,
                bins_all_targets,
                models_to_compare,
                uncertainty_error_pairs,
                np.arange(num_bins),
                "Cumulative error for ALL predictions, dataset " + dataset,
                save_path=save_location,
                error_scaling_factor=error_scaling_factor,
            )
            # Plot cumulative error figure for B1 only predictions
            plot_cumulative(
                colormap,
                bins_all_targets,
                models_to_compare,
                uncertainty_error_pairs,
                0,
                "Cumulative error for B1 predictions, dataset " + dataset,
                save_path=save_location,
                error_scaling_factor=error_scaling_factor,
            )

            # Plot cumulative error figure comparing B1 and ALL, for both models
            for model_type in models_to_compare:
                plot_cumulative(
                    colormap,
                    bins_all_targets,
                    [model_type],
                    uncertainty_error_pairs,
                    0,
                    model_type + ". Cumulative error comparing ALL and B1, dataset " + dataset,
                    compare_to_all=True,
                    save_path=save_location,
                    error_scaling_factor=error_scaling_factor,
                )

        # Set x_axis labels for following plots.
        x_axis_labels = [r"$B_{{{}}}$".format(num_bins_display + 1 - (i + 1)) for i in range(num_bins_display + 1)]

        # get error bounds
        if display_settings["errors"]:
            # mean error concat for each bin
            logger.info("mean error concat all L")
            if save_figures_bool:
                if samples_as_dots_bool:
                    filename_suffix = "_dotted"
                else:
                    filename_suffix = "_undotted"
                save_location = os.path.join(
                    save_folder, save_file_preamble + filename_suffix + "_error_all_targets.pdf"
                )

            # Create data and config objects for new API
            error_all_per_model_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_bins_concat_targets_nosep_error],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                x_label="Uncertainty Thresholded Bin",
                y_label="Localization Error (mm)",
                convert_to_percent=False,
                colormap=colormap,
                show_sample_info=show_sample_info_mode,
                show_individual_dots=samples_as_dots_bool,
                y_lim_top=box_plot_error_lim,
                to_log=True,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_per_model_boxplot(error_all_per_model_plot_data, config)

            if show_individual_target_plots:
                # plot the concatentated errors for each target seperately
                for target_idx, target_data in enumerate(all_bins_concat_targets_sep_all_error):
                    if target_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder,
                                save_file_preamble + filename_suffix + "_error_target_" + str(target_idx) + ".pdf",
                            )

                        logger.info("individual error for T%s", target_idx)

                        # Create data and config objects for new API
                        error_target_per_model_plot_data = create_boxplot_data(
                            evaluation_data_by_bins=[target_data],
                            uncertainty_categories=uncertainty_error_pairs,
                            models=models_to_compare,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Error (mm)",
                            convert_to_percent=False,
                            show_sample_info=show_sample_info_mode,
                            show_individual_dots=samples_as_dots_bool,
                            y_lim_top=box_plot_error_lim,
                            to_log=True,
                            use_list_comp=True,
                            save_path=save_location,
                            hatch_type=hatch,
                        )

                        plot_per_model_boxplot(error_target_per_model_plot_data, config)

            logger.info("Mean error")

            if save_figures_bool:
                save_location = os.path.join(
                    save_folder, save_file_preamble + filename_suffix + "_mean_error_folds_all_targets.pdf"
                )

            # Create data and config objects for new API
            mean_error_per_model_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_error_data],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Uncertainty Thresholded Bin",
                y_label="Mean Error (mm)",
                convert_to_percent=False,
                y_lim_top=box_plot_error_lim,
                to_log=True,
                show_individual_dots=True,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_per_model_boxplot(mean_error_per_model_plot_data, config)

        # Plot Error Bound Accuracy
        if display_settings["error_bounds"]:
            logger.info(" errorbound acc for all targets.")
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_errorbound_all_targets.pdf")

            # Create data and config objects for new API
            errorbound_all_generic_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_bound_data],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Uncertainty Thresholded Bin",
                y_label="Error Bound Accuracy (%)",
                save_path=save_location,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                show_individual_dots=False,
                use_list_comp=False,
                hatch_type=hatch,
            )

            plot_generic_boxplot(errorbound_all_generic_plot_data, config)

            if show_individual_target_plots:
                # plot the concatentated error bounds for each target seperately
                for target_idx, target_data in enumerate(all_bins_concat_targets_sep_all_errorbound):
                    if target_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_errorbound_target_" + str(target_idx) + ".pdf"
                            )

                        logger.info("individual errorbound acc for T%s", target_idx)

                        # Create data and config objects for new API
                        errorbound_generic_plot_data = create_boxplot_data(
                            evaluation_data_by_bins=[target_data],
                            uncertainty_categories=uncertainty_error_pairs,
                            models=models_to_compare,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Error Bound Accuracy (%)",
                            save_path=save_location,
                            y_lim_top=120,
                            width=0.2,
                            y_lim_bottom=-2,
                            font_size_label=30,
                            font_size_tick=30,
                            show_individual_dots=False,
                            use_list_comp=False,
                            hatch_type=hatch,
                        )

                        plot_generic_boxplot(errorbound_generic_plot_data, config)

        # Plot Jaccard Index
        if display_settings["jaccard"]:
            logger.info("Plot jaccard for all targets.")
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_jaccard_all_targets.pdf")

            # Create data and config objects for new API
            jaccard_all_generic_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_jaccard_data],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Uncertainty Thresholded Bin",
                y_label="Jaccard Index (%)",
                save_path=save_location,
                y_lim_top=70,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                show_individual_dots=False,
                use_list_comp=False,
                hatch_type=hatch,
            )

            plot_generic_boxplot(jaccard_all_generic_plot_data, config)

            # mean recall for each bin
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_recall_jaccard_all_targets.pdf")

            # Create data and config objects for new API
            recall_jaccard_all_generic_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_recall_data],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Recall",
                convert_to_percent=True,
                save_path=save_location,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                show_individual_dots=False,
                use_list_comp=False,
                hatch_type=hatch,
            )

            plot_generic_boxplot(recall_jaccard_all_generic_plot_data, config)

            # mean precision for each bin
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_precision_jaccard_all_targets.pdf")

            # Create data and config objects for new API
            precision_jaccard_all_generic_plot_data = create_boxplot_data(
                evaluation_data_by_bins=[all_precision_data],
                uncertainty_categories=uncertainty_error_pairs,
                models=models_to_compare,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Precision",
                convert_to_percent=True,
                save_path=save_location,
                y_lim_top=120,
                width=0.2,
                y_lim_bottom=-2,
                font_size_label=30,
                font_size_tick=30,
                show_individual_dots=False,
                use_list_comp=False,
                hatch_type=hatch,
            )

            plot_generic_boxplot(precision_jaccard_all_generic_plot_data, config)

            if show_individual_target_plots:
                # plot the jaccard index for each target seperately

                for idx_l, target_data in enumerate(all_bins_concat_targets_sep_all_jacc):
                    if idx_l in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_jaccard_target_" + str(idx_l) + ".pdf"
                            )

                        logger.info("individual jaccard for T%s", idx_l)

                        # Create data and config objects for new API
                        jaccard_generic_plot_data = create_boxplot_data(
                            evaluation_data_by_bins=[target_data],
                            uncertainty_categories=uncertainty_error_pairs,
                            models=models_to_compare,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Jaccard Index (%)",
                            save_path=save_location,
                            y_lim_top=70,
                            width=0.2,
                            y_lim_bottom=-2,
                            font_size_label=30,
                            font_size_tick=30,
                            show_individual_dots=False,
                            use_list_comp=False,
                            hatch_type=hatch,
                        )

                        plot_generic_boxplot(jaccard_generic_plot_data, config)


def generate_fig_comparing_bins(
    data: Tuple,
    display_settings: Dict[str, Any],
) -> None:
    """
    Generate figures comparing localization error, error bounds accuracy, and Jaccard index for different binning
    configurations.

    Args:
        data (Tuple): A tuple containing various inputs needed to generate the figures. The tuple should include the following elements:
            - uncertainty_error_pair (Tuple[float, float]): A tuple representing the mean and standard deviation of
            the noise uncertainty used during training and evaluation.
            - model (str): The name of the model being evaluated.
            - dataset (str): The name of the dataset being used.
            - targets (List[int]): A list of target indices being evaluated.
            - all_values_q (List[int]): A list of integers representing the number of bins being used for each evaluation.
            - colormap (str): The name of the colormap to use for plotting.
            - all_fitted_save_paths (List[str]): A list of file paths where the binned data is stored.
            - save_folder (str): The directory where the figures should be saved.
            - save_file_preamble (str): The prefix to use for all figure file names.
            - combine_middle_bins (bool): Whether to combine the middle bins or not.
            - save_figures_bool (bool): Whether to save the generated figures or not. If false, shows instead.
            - samples_as_dots_bool (bool): Whether to show individual samples as dots in the box plots or not.
            - show_sample_info_mode (str): The mode for showing sample information in the box plots.
            - box_plot_error_lim (float): The y-axis limit for the error box plots.
            - show_individual_target_plots (bool): Whether to generate individual plots for each target.
            - interpret (bool): Whether the results are being interpreted.
            - num_folds (int): The number of cross-validation folds to use.
            - ind_targets_to_show (List[int]): A list of target indices to show in individual plots.
            - error_scaling_factor (float, optional): Scaling factor for error. Defaults to 1.0.

        display_settings: Dictionary containing the following keys:
            - 'hatch': String representing the type of hatch pattern to use in the plots.

    Returns:
        None.
    """

    # Unpack data and logging settings
    [
        uncertainty_error_pair,
        model,
        dataset,
        targets,
        all_values_q,
        colormap,
        all_fitted_save_paths,
        save_folder,
        save_file_preamble,
        combine_middle_bins,  # cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"]
        save_figures_bool,  # cfg["OUTPUT"]["SAVE_FIGURES"]
        samples_as_dots_bool,  # cfg["BOXPLOT"]["SAMPLES_AS_DOTS"]
        show_sample_info_mode,  # cfg["BOXPLOT"]["SHOW_SAMPLE_INFO_MODE"]
        box_plot_error_lim,  # cfg["BOXPLOT"]["ERROR_LIM"]
        show_individual_target_plots,
        interpret,
        num_folds,
        ind_targets_to_show,
        error_scaling_factor,
    ] = data

    logger = logging.getLogger("qbin")

    hatch = display_settings["hatch"]

    # increse dimension of these for compatibility with future methods
    model_list = [model]
    uncertainty_error_pair_list = [uncertainty_error_pair]

    # If combining the middle bins we just have the 2 edge bins, and the combined middle ones.

    all_error_data = []
    all_error_target_sep = []
    all_bins_concat_targets_nosep_error = []
    all_bins_concat_targets_sep_foldwise_error = []
    all_bins_concat_targets_sep_all_error = []
    all_jaccard_data = []
    all_recall_data = []
    all_precision_data = []
    all_bins_concat_targets_sep_foldwise_jacc = []
    all_bins_concat_targets_sep_all_jacc = []
    all_bound_data = []
    all_bins_concat_targets_sep_foldwise_errorbound = []
    all_bins_concat_targets_sep_all_errorbound = []

    for idx, num_bins in enumerate(all_values_q):
        saved_bins_path_pre = all_fitted_save_paths[idx]

        bins_all_targets, bins_targets_sep, bounds_all_targets, bounds_targets_sep = generate_struct_for_qbin(
            model_list, targets, saved_bins_path_pre, dataset
        )

        # Get mean errors bin-wise, get all errors concatenated together bin-wise, and seperate by target.
        all_error_data_dict = get_mean_errors(
            bins_all_targets,
            uncertainty_error_pair_list,
            num_bins,
            targets,
            num_folds=num_folds,
            error_scaling_factor=error_scaling_factor,
            combine_middle_bins=combine_middle_bins,
        )
        all_error_data.append(all_error_data_dict["all mean error bins nosep"])
        all_error_target_sep.append(all_error_data_dict["all mean error bins targets sep"])

        all_bins_concat_targets_nosep_error.append(
            all_error_data_dict["all error concat bins targets nosep"]
        )  # shape is [num bins]
        all_bins_concat_targets_sep_foldwise_error.append(
            all_error_data_dict["all error concat bins targets sep foldwise"]
        )  # shape is [num targets][num bins]
        all_bins_concat_targets_sep_all_error.append(
            all_error_data_dict["all error concat bins targets sep all"]
        )  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

        all_jaccard_data_dict = evaluate_jaccard(
            bins_all_targets,
            uncertainty_error_pair_list,
            num_bins,
            targets,
            num_folds=num_folds,
            combine_middle_bins=combine_middle_bins,
        )
        all_jaccard_data.append(all_jaccard_data_dict["Jaccard All"])
        all_recall_data.append(all_jaccard_data_dict["Recall All"])
        all_precision_data.append(all_jaccard_data_dict["Precision All"])
        all_bins_concat_targets_sep_foldwise_jacc.append(
            all_jaccard_data_dict["all jacc concat bins targets sep foldwise"]
        )  # shape is [num targets][num bins]
        all_bins_concat_targets_sep_all_jacc.append(
            all_jaccard_data_dict["all jacc concat bins targets sep all"]
        )  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

        bound_return_dict = evaluate_bounds(
            bounds_all_targets,
            bins_all_targets,
            uncertainty_error_pair_list,
            num_bins,
            targets,
            num_folds,
            combine_middle_bins=combine_middle_bins,
        )

        all_bound_data.append(bound_return_dict["Error Bounds All"])
        all_bins_concat_targets_sep_foldwise_errorbound.append(
            bound_return_dict["all errorbound concat bins targets sep foldwise"]
        )  # shape is [num targets][num bins]
        all_bins_concat_targets_sep_all_errorbound.append(
            bound_return_dict["all errorbound concat bins targets sep all"]
        )  # same as all_bins_concat_targets_sep_foldwise but folds are flattened to a single list

    if interpret:
        # If we have combined the middle bins, we are only displaying 3 bins (outer edges, and combined middle bins).
        if combine_middle_bins:
            num_bins_display = 3
        else:
            num_bins_display = num_bins

        # Set x_axis labels for following plots.
        x_axis_labels = [str(x) for x in all_values_q]
        save_location = None

        # get error bounds

        if display_settings["errors"]:
            # mean error concat for each bin
            logger.info("mean error concat all L")
            if save_figures_bool:
                if samples_as_dots_bool:
                    filename_suffix = "_dotted"
                else:
                    filename_suffix = "_undotted"
                save_location = os.path.join(
                    save_folder, save_file_preamble + filename_suffix + "_error_all_targets.pdf"
                )

            # Create data and config objects for new API
            error_all_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_bins_concat_targets_nosep_error,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Localization Error (mm)",
                convert_to_percent=False,
                show_sample_info=show_sample_info_mode,
                show_individual_dots=samples_as_dots_bool,
                y_lim_top=box_plot_error_lim,
                to_log=True,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(error_all_plot_data, config)

            if show_individual_target_plots:
                # plot the concatentated errors for each target seperately. Must transpose the iteration.
                for target_idx in targets:
                    target_data = [x[target_idx] for x in all_bins_concat_targets_sep_all_error]

                    if target_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder,
                                save_file_preamble + filename_suffix + "_error_target_" + str(target_idx) + ".pdf",
                            )

                        logger.info("individual error for T%s", target_idx)
                        # Create data and config objects for new API
                        individual_error_plot_data = create_boxplot_data(
                            uncertainty_categories=uncertainty_error_pair_list,
                            models=model_list,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                            evaluation_data_by_bins=target_data,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Q (# Bins)",
                            y_label="Localization Error (mm)",
                            convert_to_percent=False,
                            show_sample_info=show_sample_info_mode,
                            show_individual_dots=samples_as_dots_bool,
                            y_lim_top=box_plot_error_lim,
                            to_log=True,
                            use_list_comp=True,
                            save_path=save_location,
                            hatch_type=hatch,
                        )

                        plot_comparing_q_boxplot(individual_error_plot_data, config)

            if save_figures_bool:
                save_location = os.path.join(
                    save_folder, save_file_preamble + filename_suffix + "_mean_error_folds_all_targets.pdf"
                )
            # Create data and config objects for new API
            mean_error_folds_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_error_data,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Mean Error (mm)",
                convert_to_percent=False,
                show_sample_info="None",
                show_individual_dots=False,
                y_lim_top=box_plot_error_lim,
                to_log=True,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(mean_error_folds_plot_data, config)

        # Plot Error Bound Accuracy

        if display_settings["error_bounds"]:
            logger.info(" errorbound acc for all targets.")
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_errorbound_all_targets.pdf")

            # Create data and config objects for new API
            error_bound_all_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_bound_data,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Error Bound Accuracy (%)",
                convert_to_percent=True,
                show_sample_info="None",
                show_individual_dots=False,
                y_lim_top=100,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(error_bound_all_plot_data, config)

            if show_individual_target_plots:
                # plot the concatentated errors for each target seperately. Must transpose the iteration.
                for target_idx in targets:
                    target_data = [x[target_idx] for x in all_bins_concat_targets_sep_all_errorbound]

                    if target_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_errorbound_target_" + str(target_idx) + ".pdf"
                            )

                        logger.info("individual errorbound acc for T%s", target_idx)
                        # Create data and config objects for new API
                        individual_errorbound_plot_data = create_boxplot_data(
                            uncertainty_categories=uncertainty_error_pair_list,
                            models=model_list,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                            evaluation_data_by_bins=target_data,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Q (# Bins)",
                            y_label="Error Bound Accuracy (%)",
                            convert_to_percent=True,
                            show_individual_dots=False,
                            y_lim_top=100,
                            use_list_comp=True,
                            save_path=save_location,
                            hatch_type=hatch,
                        )

                        plot_comparing_q_boxplot(individual_errorbound_plot_data, config)

        # Plot Jaccard Index
        if display_settings["jaccard"]:
            logger.info("Plot jaccard for all targets.")
            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_jaccard_all_targets.pdf")

            # Create data and config objects for new API
            jaccard_all_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_jaccard_data,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Jaccard Index (%)",
                convert_to_percent=True,
                show_individual_dots=False,
                y_lim_top=70,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(jaccard_all_plot_data, config)

            # mean recall for each bin
            logger.info("Plot recall for all targets.")

            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_recall_jaccard_all_targets.pdf")
            # Create data and config objects for new API
            recall_jaccard_all_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_recall_data,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bin Recall (%)",
                convert_to_percent=True,
                show_individual_dots=False,
                y_lim_top=120,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(recall_jaccard_all_plot_data, config)

            # mean precision for each bin
            logger.info("Plot precision for all targets.")

            if save_figures_bool:
                save_location = os.path.join(save_folder, save_file_preamble + "_precision_jaccard_all_targets.pdf")
            # Create data and config objects for new API
            precision_jaccard_all_plot_data = create_boxplot_data(
                uncertainty_categories=uncertainty_error_pair_list,
                models=model_list,
                category_labels=x_axis_labels,
                num_bins=num_bins_display,
                evaluation_data_by_bins=all_precision_data,
            )

            config = create_boxplot_config(
                colormap=colormap,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bin Precision (%)",
                convert_to_percent=True,
                show_individual_dots=False,
                y_lim_top=120,
                use_list_comp=True,
                save_path=save_location,
                hatch_type=hatch,
            )

            plot_comparing_q_boxplot(precision_jaccard_all_plot_data, config)

            if show_individual_target_plots:
                # plot the concatentated errors for each target seperately. Must transpose the iteration.
                for target_idx in targets:
                    target_data = [x[target_idx] for x in all_bins_concat_targets_sep_all_jacc]

                    if target_idx in ind_targets_to_show or ind_targets_to_show == [-1]:
                        if save_figures_bool:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_jaccard_target_" + str(target_idx) + ".pdf"
                            )

                        logger.info("individual jaccard for T%s", target_idx)
                        # Create data and config objects for new API
                        individual_jaccard_plot_data = create_boxplot_data(
                            uncertainty_categories=uncertainty_error_pair_list,
                            models=model_list,
                            category_labels=x_axis_labels,
                            num_bins=num_bins_display,
                            evaluation_data_by_bins=target_data,
                        )

                        config = create_boxplot_config(
                            colormap=colormap,
                            x_label="Q (# Bins)",
                            y_label="Jaccard Index (%)",
                            convert_to_percent=True,
                            show_individual_dots=False,
                            y_lim_top=70,
                            use_list_comp=True,
                            save_path=save_location,
                            hatch_type=hatch,
                        )

                        plot_comparing_q_boxplot(individual_jaccard_plot_data, config)

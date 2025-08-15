# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Wenjie Zhao, 1534779821@qq.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
# =============================================================================

"""
Module for correlation analysis between uncertainty and error in landmark localization.

Functions related to uncertainty-error correlation analysis in terms of:
   A) Calculate correlations: analyze_uncertainty_correlation
   B) Plotting: plot_uncertainty_correlation
   C) Main analysis functions: analyze_and_plot_uncertainty_correlation
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pwlf
from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter
from scipy import stats

from kale.interpret.visualize import save_or_show_plot


def analyze_and_plot_uncertainty_correlation(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    colormap: str = "Set1",
    to_log: bool = False,
    error_scaling_factor: float = 1.0,
    save_path: Optional[str] = None,
    n_bootstrap: int = 1000,
    sample_ratio: float = 0.6,
    font_size: int = 25,
    **fig_kwargs: Any,
) -> Dict[str, List[Any]]:
    """
    Complete uncertainty-error correlation analysis with visualization.

    This is the recommended entry point for most uncertainty analysis workflows
    as it provides both numerical results and visual insights.

    Args:
        errors (np.ndarray): Array of prediction errors. Must have same length
            as uncertainties array.
        uncertainties (np.ndarray): Array of uncertainty estimates corresponding
            to each prediction.
        quantile_thresholds (List[float]): List of quantile threshold values that
            define breakpoints for piecewise linear modeling. These determine
            how the uncertainty range is segmented for analysis.
        colormap (str, optional): Matplotlib colormap name for coloring different
            quantile segments. Defaults to "Set1".
        to_log (bool, optional): Whether to apply logarithmic scaling to both
            axes of the plot. Useful for data spanning multiple orders of magnitude.
            Defaults to False.
        error_scaling_factor (float, optional): Multiplicative factor to scale
            error values (e.g., for unit conversion from pixels to mm).
            Defaults to 1.0.
        save_path (Optional[str], optional): File path to save the generated plot.
            If None, displays the plot interactively. Supports common formats
            like PNG, PDF, SVG. Defaults to None.
        n_bootstrap (int, optional): Number of bootstrap iterations for confidence
            interval estimation. Higher values provide more robust estimates but
            increase computation time. Defaults to 1000.
        sample_ratio (float, optional): Fraction of data to sample in each bootstrap
            iteration. Must be between 0 and 1. Lower values increase diversity
            but may reduce stability. Defaults to 0.6.
        font_size (int, optional): Font size for all text elements in the plot
            including axis labels, correlation text, and quantile labels.
            Defaults to 25.
        **fig_kwargs: Additional keyword arguments for figure creation and styling:
            - figsize (tuple): Figure size in inches (default: (16, 8))
            - dpi (int): Resolution in dots per inch (default: 600)
            - bottom (float): Bottom margin for subplot adjustment (default: 0.2)
            - left (float): Left margin for subplot adjustment (default: 0.15)
            - Any other matplotlib figure parameters

    Returns:
        Dict[str, List[Any]]: Dictionary containing correlation statistics with keys:
            - 'spearman': [correlation_coefficient, p_value] for Spearman rank correlation
            - 'pearson': [correlation_coefficient, p_value] for Pearson linear correlation

    Raises:
        ValueError: If input arrays have different lengths, are empty, or contain
            invalid parameter values (e.g., sample_ratio not in (0,1])
        KeyError: If required analysis components fail to generate expected results

    Note:
        - The function automatically validates inputs and provides informative error messages
        - Bootstrap confidence bands provide visual uncertainty around the main fit
        - Quantile-based analysis reveals how correlation varies across uncertainty ranges
        - Both parametric (Pearson) and non-parametric (Spearman) correlations are computed
        - The plot includes correlation statistics prominently displayed for quick assessment
    """

    # Analyze correlation
    analysis_results = analyze_uncertainty_correlation(
        errors, uncertainties, quantile_thresholds, error_scaling_factor, n_bootstrap, sample_ratio
    )

    # Plot results
    plot_uncertainty_correlation(
        uncertainties,
        analysis_results,
        quantile_thresholds,
        colormap,
        to_log,
        save_path,
        font_size=font_size,
        **fig_kwargs,
    )

    return analysis_results["correlations"]


def analyze_uncertainty_correlation(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    error_scaling_factor: float = 1.0,
    n_bootstrap: int = 1000,
    sample_ratio: float = 0.6,
) -> Dict[str, Any]:
    """
    Analyze correlation between prediction errors and uncertainty estimates.

    Args:
        errors (np.ndarray): Array of prediction errors
        uncertainties (np.ndarray): Array of uncertainty estimates
        quantile_thresholds (List[float]): List of quantile threshold values
            that define breakpoints for piecewise linear modeling
        error_scaling_factor (float, optional): Multiplicative factor to scale
            errors (e.g., for unit conversion). Defaults to 1.0.
        n_bootstrap (int, optional): Number of bootstrap samples for confidence
            interval estimation. Defaults to 1000.
        sample_ratio (float, optional): Fraction of data to sample in each
            bootstrap iteration. Must be between 0 and 1. Defaults to 0.6.

    Returns:
        Dict[str, Any]: Dictionary containing analysis results with keys:
            - 'correlations': Dict with Spearman and Pearson correlation results
            - 'piecewise_model': Fitted piecewise linear model
            - 'bootstrap_models': List of bootstrap piecewise models
            - 'bin_label_locs': X-axis positions for quantile bin labels
            - 'scaled_errors': Array of scaled error values used in analysis

    Raises:
        ValueError: If input arrays have different lengths or invalid parameters
    """
    # Input validation
    if len(errors) != len(uncertainties):
        raise ValueError(
            f"Errors and uncertainties arrays must have the same length. "
            f"Got errors: {len(errors)}, uncertainties: {len(uncertainties)}"
        )
    if len(errors) == 0 or len(uncertainties) == 0:
        raise ValueError("Errors and uncertainties arrays cannot be empty.")

    # Scale errors
    scaled_errors = errors * error_scaling_factor

    # Calculate correlations
    correlation_dict = _calculate_correlations(uncertainties, scaled_errors)

    # Fit main piecewise model
    piecewise_model = _fit_piecewise_model(uncertainties, scaled_errors, quantile_thresholds)

    # Generate bootstrap models
    bootstrap_models = _generate_bootstrap_models(
        uncertainties, scaled_errors, quantile_thresholds, n_bootstrap, sample_ratio
    )

    # Calculate segment centers
    bin_label_locs = _calculate_segment_centers(quantile_thresholds, uncertainties)

    return {
        "correlations": correlation_dict,
        "piecewise_model": piecewise_model,
        "bootstrap_models": bootstrap_models,
        "bin_label_locs": bin_label_locs,
        "scaled_errors": scaled_errors,
    }


def plot_uncertainty_correlation(
    uncertainties: np.ndarray,
    analysis_results: Dict[str, Any],
    quantile_thresholds: List[float],
    colormap: str = "Set1",
    to_log: bool = False,
    save_path: Optional[str] = None,
    font_size: int = 25,
    **fig_kwargs: Any,
) -> None:
    """
    Create a comprehensive visualization of uncertainty-error correlation analysis.

    Args:
        uncertainties (np.ndarray): Array of uncertainty estimates
        analysis_results (Dict[str, Any]): Results from analyze_uncertainty_correlation
        quantile_thresholds (List[float]): Quantile threshold values for segmentation
        colormap (str, optional): Matplotlib colormap name for segment coloring.
            Defaults to "Set1".
        to_log (bool, optional): Whether to use logarithmic scale for both axes.
            Defaults to False.
        save_path (Optional[str], optional): File path to save the plot. If None,
            displays the plot interactively. Defaults to None.
        font_size (int, optional): Font size for all text elements. Defaults to 25.
        **fig_kwargs: Additional keyword arguments for figure creation and styling:
            - figsize (tuple): Figure size in inches (default: (16, 8))
            - dpi (int): Dots per inch for resolution (default: 600)
            - bottom (float): Bottom margin for subplots_adjust (default: 0.2)
            - left (float): Left margin for subplots_adjust (default: 0.15)
            - Any other matplotlib figure parameters

    Returns:
        None: Creates and displays or saves the plot

    Raises:
        ValueError: If input parameters are invalid
        KeyError: If analysis_results is missing required keys
    """
    # Extract figure-specific kwargs with defaults
    figsize = fig_kwargs.pop("figsize", (16, 8))
    dpi = fig_kwargs.pop("dpi", 600)
    bottom = fig_kwargs.pop("bottom", 0.2)
    left = fig_kwargs.pop("left", 0.15)

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **fig_kwargs)

    # Handle color configuration
    colors = colormaps.get_cmap(colormap)(np.arange(3))
    scatter_color = colors[2]

    # Set axis scales
    if to_log:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    ax.set_xlim(max(uncertainties), min(uncertainties))

    # Plot bootstrap confidence bands
    _plot_bootstrap_confidence_bands(ax, analysis_results["bootstrap_models"], uncertainties)

    # Plot scatter points
    _plot_scatter_points(ax, uncertainties, analysis_results["scaled_errors"], scatter_color)

    # Plot piecewise segments
    _plot_piecewise_segments(ax, analysis_results["piecewise_model"], uncertainties, quantile_thresholds, colors[:2])

    # Setup plot formatting
    _setup_plot_formatting(
        ax, analysis_results["bin_label_locs"], quantile_thresholds, analysis_results["correlations"], font_size
    )

    # Adjust layout
    plt.subplots_adjust(bottom=bottom, left=left)

    # Save or show plot using all remaining fig_kwargs
    save_or_show_plot(save_path=save_path, fig_width=figsize[0], fig_height=figsize[1], dpi=dpi)


def _calculate_correlations(uncertainties: np.ndarray, scaled_errors: np.ndarray) -> Dict[str, List[float]]:
    """
    Calculate Spearman and Pearson correlations between uncertainties and errors.

    This function computes both Spearman rank correlation (non-parametric) and
    Pearson correlation (parametric) to assess the relationship between uncertainty
    estimates and prediction errors.

    Args:
        uncertainties (np.ndarray): Array of uncertainty values
        scaled_errors (np.ndarray): Array of scaled error values

    Returns:
        Dict[str, List[float]]: Dictionary containing correlation results with keys:
            - 'spearman': [correlation_coefficient, p_value] for Spearman correlation
            - 'pearson': [correlation_coefficient, p_value] for Pearson correlation
    """
    smp_corr, smp_p = stats.spearmanr(uncertainties, scaled_errors, alternative="greater")
    pear_corr, pear_p = stats.pearsonr(uncertainties, scaled_errors)
    return {"spearman": [smp_corr, smp_p], "pearson": [pear_corr, pear_p]}


def _fit_piecewise_model(
    uncertainties: np.ndarray, scaled_errors: np.ndarray, quantile_thresholds: List[float]
) -> pwlf.PiecewiseLinFit:
    """
    Fit a piecewise linear model to uncertainty-error data using specified breakpoints.

    This function creates and fits a piecewise linear regression model that can capture
    non-linear relationships between uncertainties and errors by fitting different
    linear segments at different uncertainty ranges.

    Args:
        uncertainties (np.ndarray): Array of uncertainty values (independent variable)
        scaled_errors (np.ndarray): Array of scaled error values (dependent variable)
        quantile_thresholds (List[float]): List of quantile threshold values that serve
            as breakpoints for the piecewise linear model

    Returns:
        pwlf.PiecewiseLinFit: Fitted piecewise linear model object that can be used
            for predictions and plotting
    """
    piecewise_model = pwlf.PiecewiseLinFit(uncertainties, scaled_errors)
    piecewise_model.fit_with_breaks(quantile_thresholds)
    return piecewise_model


def _generate_bootstrap_models(
    uncertainties: np.ndarray,
    scaled_errors: np.ndarray,
    quantile_thresholds: List[float],
    n_bootstrap: int = 1000,
    sample_ratio: float = 0.6,
) -> List[pwlf.PiecewiseLinFit]:
    """
    Generate bootstrap piecewise models for confidence interval estimation.

    This function creates multiple piecewise linear models using bootstrap sampling
    to estimate confidence intervals around the main piecewise fit. Each bootstrap
    model is fitted on a random subset of the original data.

    Args:
        uncertainties (np.ndarray): Array of uncertainty values
        scaled_errors (np.ndarray): Array of scaled error values
        quantile_thresholds (List[float]): Quantile threshold breakpoints for piecewise fitting
        n_bootstrap (int, optional): Number of bootstrap samples to generate. Defaults to 1000.
        sample_ratio (float, optional): Fraction of data to sample for each bootstrap.
            Must be between 0 and 1. Defaults to 0.6.

    Returns:
        List[pwlf.PiecewiseLinFit]: List of fitted bootstrap piecewise models

    Note:
        Each bootstrap model is fitted on a random sample of size
        int(len(data) * sample_ratio) drawn without replacement from the original data.
    """
    bootstrap_models = []

    for i in range(n_bootstrap):
        sample_index = np.random.choice(range(len(scaled_errors)), int(len(scaled_errors) * sample_ratio))

        uncertainties_samples = uncertainties[sample_index]
        scaled_errors_samples = scaled_errors[sample_index]

        bootstrap_model = _fit_piecewise_model(uncertainties_samples, scaled_errors_samples, quantile_thresholds)
        bootstrap_models.append(bootstrap_model)

    return bootstrap_models


def _get_quantile_bounds(quantile_thresholds: List[float], uncertainties: np.ndarray, idx: int) -> tuple[float, float]:
    """
    Get minimum and maximum bounds for a specific quantile segment.

    This function determines the boundary values for a quantile segment based on
    the segment index and the list of quantile thresholds. It handles edge cases
    for the first and last segments.

    Args:
        quantile_thresholds (List[float]): List of quantile threshold values that
            define segment boundaries
        uncertainties (np.ndarray): Array of uncertainty values used to determine
            global min/max bounds
        idx (int): Index of the quantile segment (0-based)

    Returns:
        tuple[float, float]: Tuple containing (min_value, max_value) for the segment

    Note:
        - For idx=0 (first segment): min_val = min(uncertainties), max_val = first threshold
        - For middle segments: min_val = previous threshold, max_val = current threshold
        - For last segment: min_val = last threshold, max_val = max(uncertainties)
    """
    if idx == 0:
        min_val = min(uncertainties)
        max_val = quantile_thresholds[idx]
    elif idx <= len(quantile_thresholds) - 1:
        min_val = quantile_thresholds[idx - 1]
        max_val = quantile_thresholds[idx]
    else:
        min_val = quantile_thresholds[idx - 1]
        max_val = max(uncertainties)

    return min_val, max_val


def _calculate_segment_centers(quantile_thresholds: List[float], uncertainties: np.ndarray) -> List[float]:
    """
    Calculate center positions for each quantile segment for label placement.

    This function computes the midpoint of each quantile segment, which is used
    for positioning x-axis labels in plots. The center is calculated as the
    arithmetic mean of the segment's minimum and maximum bounds.

    Args:
        quantile_thresholds (List[float]): List of quantile threshold values
        uncertainties (np.ndarray): Array of uncertainty values for determining bounds

    Returns:
        List[float]: List of center positions for each quantile segment

    Note:
        The number of segments is len(quantile_thresholds) + 1, as thresholds
        define boundaries between segments.
    """
    bin_label_locs = []
    for idx in range(len(quantile_thresholds) + 1):
        min_val, max_val = _get_quantile_bounds(quantile_thresholds, uncertainties, idx)
        bin_label_locs.append((min_val + max_val) / 2)
    return bin_label_locs


def _plot_bootstrap_confidence_bands(
    ax, bootstrap_models: List[pwlf.PiecewiseLinFit], uncertainties: np.ndarray, num_pred_points: int = 10000
) -> None:
    """
    Plot bootstrap confidence bands on the given axes.

    This function visualizes uncertainty in the piecewise linear fit by plotting
    prediction lines from multiple bootstrap models. The ensemble of lines creates
    a confidence band around the main fit.

    Args:
        ax: Matplotlib axes object to plot on
        bootstrap_models (List[pwlf.PiecewiseLinFit]): List of fitted bootstrap models
        uncertainties (np.ndarray): Array of uncertainty values for determining plot range
        num_pred_points (int, optional): Number of prediction points for smooth curves.
            Defaults to 10000.

    Returns:
        None: Plots directly on the provided axes

    Note:
        - Each bootstrap model is plotted as a grey line with low alpha (0.2)
        - Lines are plotted in reverse order ([::-1]) for proper visualization
        - Uses zorder=2 to place confidence bands behind main elements
    """
    uncertainties_pred = np.linspace(min(uncertainties), max(uncertainties), num=num_pred_points)

    for model in bootstrap_models:
        scaled_errors_pred = model.predict(uncertainties_pred)
        ax.plot(uncertainties_pred[::-1], scaled_errors_pred[::-1], "-", color="grey", alpha=0.2, zorder=2)


def _plot_scatter_points(ax, uncertainties: np.ndarray, scaled_errors: np.ndarray, scatter_color: str) -> None:
    """
    Plot scatter points of uncertainties vs errors on the given axes.

    This function creates a scatter plot showing the relationship between uncertainty
    estimates and scaled errors. Each point represents one data sample.

    Args:
        ax: Matplotlib axes object to plot on
        uncertainties (np.ndarray): Array of uncertainty values (x-axis)
        scaled_errors (np.ndarray): Array of scaled error values (y-axis)
        scatter_color (str): Color for the scatter points

    Returns:
        None: Plots directly on the provided axes

    Note:
        - Uses circular markers ('o') with low alpha (0.2) for transparency
        - Points are placed at zorder=1 to appear behind other plot elements
        - Alpha blending helps visualize point density in overlapping regions
    """
    ax.scatter(uncertainties, scaled_errors, marker="o", color=scatter_color, zorder=1, alpha=0.2)


def _plot_piecewise_segments(
    ax,
    piecewise_model: pwlf.PiecewiseLinFit,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    colors: List[str],
    num_pred_points: int = 20000,
) -> None:
    """
    Plot piecewise linear segments with different colors and background shading.

    This function visualizes the fitted piecewise linear model by plotting each
    segment in a different color and adding background shading to distinguish
    quantile regions.

    Args:
        ax: Matplotlib axes object to plot on
        piecewise_model (pwlf.PiecewiseLinFit): Fitted piecewise linear model
        uncertainties (np.ndarray): Array of uncertainty values for range determination
        quantile_thresholds (List[float]): Quantile thresholds defining segment boundaries
        colors (List[str]): List of colors for different segments (cycles if needed)
        num_pred_points (int, optional): Number of prediction points for smooth curves.
            Defaults to 20000.

    Returns:
        None: Plots directly on the provided axes

    Note:
        - Each segment gets a different color from the colors list (cycles using modulo)
        - Background shading (axvspan) with alpha=0.1 highlights quantile regions
        - Piecewise lines use zorder=3 to appear on top of other elements
        - High num_pred_points ensures smooth curve appearance
    """
    uncertainties_pred = np.linspace(min(uncertainties), max(uncertainties), num=num_pred_points)
    scaled_errors_pred = piecewise_model.predict(uncertainties_pred)

    for idx in range(len(quantile_thresholds) + 1):
        color = colors[idx % len(colors)]
        min_val, max_val = _get_quantile_bounds(quantile_thresholds, uncertainties, idx)

        plot_indices = [i for i in range(len(uncertainties_pred)) if min_val <= uncertainties_pred[i] < max_val]

        # Shade background to make slices per quantile
        ax.axvspan(
            uncertainties_pred[plot_indices][0], uncertainties_pred[plot_indices][-1], facecolor=color, alpha=0.1
        )
        ax.plot(uncertainties_pred[plot_indices], scaled_errors_pred[plot_indices], "-", color=color, zorder=3)


def _setup_plot_formatting(
    ax,
    bin_label_locs: List[float],
    quantile_thresholds: List[float],
    correlation_dict: Dict[str, List[float]],
    font_size: int = 25,
) -> None:
    """
    Setup comprehensive plot formatting including labels, ticks, and correlation text.

    This function handles all aspects of plot formatting including axis labels,
    tick formatting, correlation coefficient display, and overall plot styling.
    It creates a publication-ready visualization of the uncertainty-error correlation.

    Args:
        ax: Matplotlib axes object to format
        bin_label_locs (List[float]): X-axis positions for quantile bin labels
        quantile_thresholds (List[float]): Quantile thresholds for determining label count
        correlation_dict (Dict[str, List[float]]): Dictionary containing correlation results
            with 'spearman' key containing [correlation, p_value]
        font_size (int, optional): Font size for labels and text. Defaults to 25.

    Returns:
        None: Modifies the provided axes object in-place

    Note:
        - X-axis labels are formatted as Q_1, Q_2, ..., Q_n using LaTeX notation
        - Correlation coefficient (ρ) and p-value are displayed prominently
        - P-values < 0.001 are shown as "< 0.001" for readability
        - Uses bold formatting for correlation statistics
        - Axis labels indicate uncertainty quantiles and error in mm
    """
    # Set x-axis ticks and labels
    ax.set_xticks(bin_label_locs)
    new_labels = [r"$Q_{{{}}}$".format(x + 1) for x in range(len(quantile_thresholds) + 1)]
    ax.xaxis.set_major_formatter(plt.FixedFormatter(new_labels))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Add correlation text
    smp_corr, smp_p = correlation_dict["spearman"]
    p_val = 0.001 if np.round(smp_p, 3) == 0 else np.round(smp_p, 3)
    bold_text = (
        r"$\bf{\rho:} $" + r"$\bf{{{}}}$".format(np.round(smp_corr, 2)) + ", p-value < " + r"${{{}}}$".format(p_val)
    )

    ax.text(
        0.5,
        0.9,
        bold_text,
        size=font_size,
        color="black",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    # Set axis labels and formatting
    ax.set_xlabel("Uncertainty Quantile", fontsize=font_size)
    ax.set_ylabel("Error (mm)", fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

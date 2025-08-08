"""
Authors: Lawrence Schobs, Zhao Wenjie, Ji Zhongwei

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


def _calculate_correlations(uncertainties: np.ndarray, scaled_errors: np.ndarray) -> Dict[str, List[float]]:
    """Calculate Spearman and Pearson correlations."""
    smp_corr, smp_p = stats.spearmanr(uncertainties, scaled_errors, alternative="greater")
    pear_corr, pear_p = stats.pearsonr(uncertainties, scaled_errors)
    return {"spearman": [smp_corr, smp_p], "pearson": [pear_corr, pear_p]}


def _fit_piecewise_model(
    uncertainties: np.ndarray, scaled_errors: np.ndarray, quantile_thresholds: List[float]
) -> pwlf.PiecewiseLinFit:
    """Fit piecewise linear model to data."""
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
    """Generate bootstrap piecewise models for confidence intervals."""
    bootstrap_models = []

    for i in range(n_bootstrap):
        sample_index = np.random.choice(range(len(scaled_errors)), int(len(scaled_errors) * sample_ratio))

        uncertainties_samples = uncertainties[sample_index]
        scaled_errors_samples = scaled_errors[sample_index]

        bootstrap_model = _fit_piecewise_model(uncertainties_samples, scaled_errors_samples, quantile_thresholds)
        bootstrap_models.append(bootstrap_model)

    return bootstrap_models


def _get_quantile_bounds(quantile_thresholds: List[float], uncertainties: np.ndarray, idx: int) -> tuple[float, float]:
    """Get min and max bounds for a quantile segment."""
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
    """Calculate center positions for each quantile segment."""
    bin_label_locs = []
    for idx in range(len(quantile_thresholds) + 1):
        min_val, max_val = _get_quantile_bounds(quantile_thresholds, uncertainties, idx)
        bin_label_locs.append((min_val + max_val) / 2)
    return bin_label_locs


def _plot_bootstrap_confidence_bands(
    ax, bootstrap_models: List[pwlf.PiecewiseLinFit], uncertainties: np.ndarray, num_pred_points: int = 10000
) -> None:
    """Plot bootstrap confidence bands."""
    uncertainties_pred = np.linspace(min(uncertainties), max(uncertainties), num=num_pred_points)

    for model in bootstrap_models:
        scaled_errors_pred = model.predict(uncertainties_pred)
        ax.plot(uncertainties_pred[::-1], scaled_errors_pred[::-1], "-", color="grey", alpha=0.2, zorder=2)


def _plot_scatter_points(ax, uncertainties: np.ndarray, scaled_errors: np.ndarray, scatter_color: str) -> None:
    """Plot scatter points of uncertainties vs errors."""
    ax.scatter(uncertainties, scaled_errors, marker="o", color=scatter_color, zorder=1, alpha=0.2)


def _plot_piecewise_segments(
    ax,
    piecewise_model: pwlf.PiecewiseLinFit,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    colors: List[str],
    num_pred_points: int = 20000,
) -> None:
    """Plot piecewise linear segments with different colors."""
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
    """Setup plot formatting including labels, ticks, and correlation text."""
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


def analyze_uncertainty_correlation(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    error_scaling_factor: float = 1.0,
    n_bootstrap: int = 1000,
    sample_ratio: float = 0.6,
) -> Dict[str, Any]:
    """
    Analyze correlation between errors and uncertainties.

    Args:
        errors: Array of errors
        uncertainties: Array of uncertainties
        quantile_thresholds: List of quantile thresholds
        error_scaling_factor: Scaling factor for errors
        n_bootstrap: Number of bootstrap samples for confidence intervals
        sample_ratio: Ratio of samples to use for each bootstrap sample

    Returns:
        Dictionary containing correlation stats and fitted models
    """
    # Input validation
    if len(errors) != len(uncertainties):
        raise ValueError(
            f"Errors and uncertainties arrays must have the same length. "
            f"Got errors: {len(errors)}, uncertainties: {len(uncertainties)}"
        )

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
    Plot uncertainty correlation analysis results.

    Args:
        uncertainties: Array of uncertainties
        analysis_results: Results from analyze_uncertainty_correlation
        quantile_thresholds: List of quantile thresholds
        colormap: Name of matplotlib colormap
        to_log: Whether to use log scale
        save_path: Path to save plot
        font_size: Font size for labels and titles
        **fig_kwargs: Additional keyword arguments for figure creation. Supported parameters:
            - figsize: tuple, figure size in inches (default: (16, 8))
            - dpi: int, dots per inch (default: 600)
            - bottom: float, bottom margin for subplots_adjust (default: 0.2)
            - left: float, left margin for subplots_adjust (default: 0.15)
            - Any other matplotlib figure parameters
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
    Calculates Spearman correlation between errors and uncertainties.
    Plots piecewise linear regression with bootstrap confidence intervals.

    This is a convenience function that combines analysis and plotting.

    Args:
        errors (np.ndarray): Array of errors.
        uncertainties (np.ndarray): Array of uncertainties.
        quantile_thresholds (List[float]): List of quantile thresholds.
        colormap (str): Name of matplotlib colormap. Defaults to 'Set1'.
        to_log (bool, optional): Whether to apply logarithmic transformation on axes. Defaults to False.
        error_scaling_factor (float, optional): Scaling factor for error. Defaults to 1.0.
        save_path (Optional[str], optional): Path to save the plot, if None, the plot will be shown. Defaults to None.
        n_bootstrap (int, optional): Number of bootstrap samples for confidence intervals. Defaults to 1000.
        sample_ratio (float, optional): Ratio of samples to use for each bootstrap sample. Defaults to 0.6.
        font_size (int, optional): Font size for plot labels and titles. Defaults to 25.
        **fig_kwargs: Additional keyword arguments for figure creation. Supported parameters:
            - figsize: tuple, figure size in inches (default: (12, 8))
            - dpi: int, dots per inch (default: 600)
            - bottom: float, bottom margin for subplots_adjust (default: 0.2)
            - left: float, left margin for subplots_adjust (default: 0.15)
            - Any other matplotlib figure parameters

    Returns:
        Dict[str, List[Any]]: Dictionary containing Spearman and Pearson correlation coefficients and p-values.
    """
    # Input validation
    if len(errors) != len(uncertainties):
        raise ValueError(
            f"Errors and uncertainties arrays must have the same length. "
            f"Got errors: {len(errors)}, uncertainties: {len(uncertainties)}"
        )

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

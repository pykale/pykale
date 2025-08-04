"""
Correlation of uncertainty with error (fit_line_with_ci)
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pwlf
from matplotlib.ticker import ScalarFormatter
from scipy import stats


def fit_line_with_ci(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    quantile_thresholds: List[float],
    cmaps: List[Dict[str, str]],
    to_log: bool = False,
    error_scaling_factor: float = 1.0,
    save_path: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """
    Calculates Spearman correlation between errors and uncertainties.
    Plots piecewise linear regression with bootstrap confidence intervals.
    Breakpoints in linear regression are defined by the uncertainty quantiles of the data.

    Args:
        errors (np.ndarray): Array of errors.
        uncertainties (np.ndarray): Array of uncertainties.
        quantile_thresholds (List[float]): List of quantile thresholds.
        cmaps (List[Dict[str, str]]): List of colormap dictionaries.
        to_log (bool, optional): Whether to apply logarithmic transformation on axes. Defaults to False.
        error_scaling_factor (float, optional): Scaling factor for error. Defaults to 1.0.
        save_path (Optional[str], optional): Path to save the plot, if None, the plot will be shown. Defaults to None.

    Returns:
        Dict[str, List[Any]]: Dictionary containing Spearman and Pearson correlation coefficients and p-values.
    """

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    if to_log:
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

    X = uncertainties
    y = errors * error_scaling_factor
    plt.xlim(max(X), min(X))

    # Calculate correlations
    spm_corr, spm_p = stats.spearmanr(X, y, alternative="greater")
    pear_corr, pear__p = stats.pearsonr(X, y)
    correlation_dict = {"spearman": [spm_corr, spm_p], "pearson": [pear_corr, pear__p]}

    # Bootstrap sample for confidence intervals
    for i in range(0, 1000):
        sample_index = np.random.choice(range(0, len(y)), int(len(y) * 0.6))

        X_samples = X[sample_index]
        y_samples = y[sample_index]
        my_pwlf = pwlf.PiecewiseLinFit(X_samples, y_samples)
        my_pwlf.fit_with_breaks(quantile_thresholds)
        xHat = np.linspace(min(X), max(X), num=10000)
        yHat = my_pwlf.predict(xHat)

        # Plot each bootstrap as a grey line
        plt.plot(xHat[::-1], yHat[::-1], "-", color="grey", alpha=0.2, zorder=2)

    # Display all datapoints on plot
    plt.scatter(X, y, marker="o", color=cmaps[2], zorder=1, alpha=0.2)

    # Final fit on ALL the data
    my_pwlf = pwlf.PiecewiseLinFit(X, y)

    # Manually set the piecewise breaks as the quantile thresholds.
    my_pwlf.fit_with_breaks(quantile_thresholds)

    # Plot the final fitted line, changing the color of the line segments for each quantile for visualisation.
    bin_label_locs = []
    for idx in range(len(quantile_thresholds) + 1):
        color = "blue" if idx % 2 == 0 else "red"
        xHat = np.linspace(min(X), max(X), num=20000)
        yHat = my_pwlf.predict(xHat)

        if idx == 0:
            min_ = min(X)
            max_ = quantile_thresholds[idx]

        elif idx <= len(quantile_thresholds) - 1:
            min_ = quantile_thresholds[idx - 1]
            max_ = quantile_thresholds[idx]
        else:
            min_ = quantile_thresholds[idx - 1]
            max_ = max(X)

        plot_indices = [i for i in range(len(xHat)) if min_ <= xHat[i] < max_]
        bin_label_locs.append((min_ + max_) / 2)
        quantile_data_indices = [i for i in range(len(X)) if min_ <= X[i] < max_]
        # shade background to make slices per quantile
        q_spm_corr, q_spm_p = stats.spearmanr(X[quantile_data_indices], y[quantile_data_indices])
        plt.axvspan(xHat[plot_indices][0], xHat[plot_indices][-1], facecolor=color, alpha=0.1)
        plt.plot(xHat[plot_indices], yHat[plot_indices], "-", color=color, zorder=3)

    ax.set_xticks(bin_label_locs)
    new_labels = [r"$Q_{{{}}}$".format(x + 1) for x in range(len(quantile_thresholds) + 1)]
    ax.xaxis.set_major_formatter(plt.FixedFormatter(new_labels))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Put text on plot showing the correlation
    p_vl = 0.001 if np.round(spm_p, 3) == 0 else np.round(spm_p, 3)
    bold_text = (
        r"$\bf{\rho:} $" + r"$\bf{{{}}}$".format(np.round(spm_corr, 2)) + ", p-value < " + r"${{{}}}$".format(p_vl)
    )

    plt.text(
        0.5,
        0.9,
        bold_text,
        size=25,
        color="black",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Uncertainty Quantile", fontsize=25)
    ax.set_ylabel("Error (mm)", fontsize=25)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.15)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if save_path is not None:
        plt.gcf().set_size_inches(16.0, 8.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.show()
        plt.close()

    return correlation_dict

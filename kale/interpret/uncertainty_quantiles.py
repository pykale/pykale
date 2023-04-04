# =============================================================================
# Author: Lawrence Schobs, laschobs1@sheffield.ac.uk
# =============================================================================

import logging
import math
import os

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pwlf
from matplotlib.ticker import ScalarFormatter
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from kale.evaluate.uncertainty_metrics import (
    evaluate_bounds,
    evaluate_correlations,
    evaluate_jaccard,
    generate_summary_df,
    get_mean_errors,
)
from kale.prepdata.tabular_transform import get_data_struct


def fit_line_with_ci(errors, uncertainties, quantile_thresholds, cmaps, to_log=False, pixel_to_mm=1.0, save_path=None):
    """Calculates spearman correlation between errors and uncertainties. Plots piecewise linear regression with bootstrap confidence intervals.
       Breakpoints in linear regression are defined by the uncertainty quantiles of the data.

    Args:
        errors (_type_): _description_
        uncertainties (_type_): _description_
        quantile_thresholds (_type_): _description_
        cmaps (_type_): _description_
        to_log (bool, optional): _description_. Defaults to False.
        pixel_to_mm (float, optional): _description_. Defaults to 1.0.
        save_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    if to_log:
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

    X = uncertainties
    y = errors * pixel_to_mm
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

    # Plot the final fitted line, changing the colour of the line segments for each quantile for visualisation.
    bin_label_locs = []
    for idx in range(len(quantile_thresholds) + 1):
        colour = "blue" if idx % 2 == 0 else "red"
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

        plot_indicies = [i for i in range(len(xHat)) if min_ <= xHat[i] < max_]
        bin_label_locs.append((min_ + max_) / 2)
        quantile_data_indicies = [i for i in range(len(X)) if min_ <= X[i] < max_]
        # shade background to make slices per quantile
        q_spm_corr, q_spm_p = stats.spearmanr(X[quantile_data_indicies], y[quantile_data_indicies])
        plt.axvspan(xHat[plot_indicies][0], xHat[plot_indicies][-1], facecolor=colour, alpha=0.1)
        plt.plot(xHat[plot_indicies], yHat[plot_indicies], "-", color=colour, zorder=3)

    # plt.xticks(bin_label_locs)
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


def quantile_binning_and_est_errors(
    errors, uncertainties, num_bins, type="quantile", acceptable_thresh=5, combine_middle_bins=False
):
    """
    Calculate quantile thresholds, and isotonically regress errors and uncertainties and get estimated error bounds.

    Args:
        errors (list): list of errors,
        uncertainties (list): list of uncertainties,
        num_bins (int): Number of quantile bins,

        type (str): what type of thresholds to calculate. quantile recommended. (default='quantile),
        acceptable_thresh (float):acceptable error threshold. only relevent if type="error-wise".


    Returns:
        [list,list]: list of quantile thresholds and estimated error bounds.
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
        # print("estimated errors before: ", estimated_errors)
        # print("\n Full estimated errors combined middle: ", estimated_errors)
        # print(" Fullestimated uncert_boundaries: ", uncert_boundaries)
        estimated_errors = [estimated_errors[0], estimated_errors[-1]]
        uncert_boundaries = [uncert_boundaries[0], uncert_boundaries[-1]]
        # print("Outer estimated errors combined middle: ", estimated_errors)
        # print("Outer estimated uncert_boundaries: ", uncert_boundaries)

    return uncert_boundaries, estimated_errors


def box_plot(
    cmaps,
    landmark_uncert_dicts,
    uncertainty_types_list,
    models,
    x_axis_labels,
    x_label,
    y_label,
    num_bins,
    show_sample_info="None",
    save_path=None,
    y_lim=120,
    turn_to_percent=True,
    to_log=False,
):
    """
    Creates a box plot of data.

    Args:
        cmaps (list): list of colours for matplotlib,
        landmark_uncert_dicts (Dict): Dict of pandas dataframe for the data to dsiplay,
        uncertainty_types_list ([list]): list of lists describing the different uncert combinations to test,
        models (list): the models we want to compare, keys in landmark_uncert_dicts,
        x_axis_labels (list): list of strings for the x-axis labels, one for each bin,
        x_label (str): x axis label,
        y_label (int): y axis label,
        num_bins (int): Number of uncertainty bins,
        save_path (str):path to save plot to. If None, displays on screen (default=None),
        y_lim (int): y axis limit of graph (default=120),


    """

    hatch_type = "o"

    plt.style.use("fivethirtyeight")

    orders = []
    ax = plt.gca()

    # fig.set_size_inches(24, 10)

    ax.xaxis.grid(False)

    bin_label_locs = []
    all_rects = []
    outer_min_x_loc = 0
    middle_min_x_loc = 0
    inner_min_x_loc = 0

    circ_patches = []

    for i, (up) in enumerate(uncertainty_types_list):
        uncertainty_type = up[0]

        for j in range(num_bins):
            inbetween_locs = []
            for hash_idx, model_type in enumerate(models):
                if j == 0:
                    if hash_idx == 1:
                        circ11 = patches.Patch(
                            facecolor=cmaps[i],
                            label=model_type + " " + uncertainty_type,
                            hatch=hatch_type,
                            edgecolor="black",
                        )
                    else:
                        circ11 = patches.Patch(facecolor=cmaps[i], label=model_type + " " + uncertainty_type)
                    circ_patches.append(circ11)

                dict_key = [
                    x for x in list(landmark_uncert_dicts.keys()) if (model_type in x) and (uncertainty_type in x)
                ][0]
                model_data = landmark_uncert_dicts[dict_key]
                all_b_data = model_data[j]

                orders.append(model_type + uncertainty_type)

                width = 0.2

                x_loc = [(outer_min_x_loc + inner_min_x_loc + middle_min_x_loc)]
                inbetween_locs.append(x_loc[0])

                # Turn data to percentages
                if turn_to_percent:
                    percent_data = [(x) * 100 for x in all_b_data]
                else:
                    percent_data = all_b_data
                rect = ax.boxplot(
                    percent_data, positions=x_loc, sym="", widths=width, showmeans=True, patch_artist=True
                )

                # Set colour, pattern, median line and mean marker.
                for r in rect["boxes"]:
                    r.set(color="black", linewidth=1)
                    r.set(facecolor=cmaps[i])

                    if hash_idx == 1:
                        r.set_hatch(hatch_type)
                for median in rect["medians"]:
                    median.set(color="crimson", linewidth=3)

                for mean in rect["means"]:
                    mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=10)

                all_rects.append(rect)

                inner_min_x_loc += 0.1 + width
            bin_label_locs.append(np.mean(inbetween_locs))
            middle_min_x_loc += 0.02

        if num_bins > 10:
            outer_min_x_loc += 0.35
        else:
            outer_min_x_loc += 0.25

    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.set_xticks(bin_label_locs)

    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    if num_bins <= 5:
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis_labels[:-1] * (len(uncertainty_types_list) * 2)))
    # If too many bins, only show the first and last or it will appear too squished, indicate direction with arrow.
    elif num_bins < 15:
        number_blanks_0 = ["" for x in range(math.floor((num_bins - 3) / 2))]
        number_blanks_1 = ["" for x in range(num_bins - 3 - len(number_blanks_0))]
        new_labels = [x_axis_labels[-0]] + number_blanks_0 + [r"$\rightarrow$"] + number_blanks_1 + [x_axis_labels[-1]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))
    # if more than 15 bins, we must move the first and last labels inwards to prevent overlap.
    else:
        number_blanks_0 = ["" for x in range(math.floor((num_bins - 5) / 2))]
        number_blanks_1 = ["" for x in range(num_bins - 5 - len(number_blanks_0))]
        new_labels = (
            [""]
            + [x_axis_labels[0]]
            + number_blanks_0
            + [r"$\rightarrow$"]
            + number_blanks_1
            + [x_axis_labels[-1]]
            + [""]
        )
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))

    if to_log:
        # ax.set_yscale("log",base=2)
        ax.set_yscale("symlog", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-2, y_lim)

    else:
        ax.set_ylim((-2, y_lim))

    # If using percent, doesnt make sense to show any y tick above 100
    if turn_to_percent and y_lim > 100:
        plt.yticks(np.arange(0, y_lim, 20))

    ax.legend(handles=circ_patches, loc=9, fontsize=30, ncol=3, columnspacing=6)

    if save_path is not None:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.show()
        plt.close()


def box_plot_per_model(
    cmaps,
    landmark_uncert_dicts,
    uncertainty_types_list,
    models,
    x_axis_labels,
    x_label,
    y_label,
    num_bins,
    show_sample_info="None",
    save_path=None,
    y_lim=120,
    turn_to_percent=True,
    to_log=False,
    show_individual_dots=True,
):
    """
    Creates a box plot of data.

    Args:
        cmaps (list): list of colours for matplotlib,
        landmark_uncert_dicts (Dict): Dict of pandas dataframe for the data to dsiplay,
        uncertainty_types_list ([list]): list of lists describing the different uncert combinations to test,
        models (list): the models we want to compare, keys in landmark_uncert_dicts,
        x_axis_labels (list): list of strings for the x-axis labels, one for each bin,
        x_label (str): x axis label,
        y_label (int): y axis label,
        num_bins (int): Number of uncertainty bins,
        save_path (str):path to save plot to. If None, displays on screen (default=None),
        y_lim (int): y axis limit of graph (default=120),


    """

    hatch_type = "o"
    logger = logging.getLogger("qbin")

    plt.style.use("fivethirtyeight")

    orders = []
    ax = plt.gca()

    # fig.set_size_inches(24, 10)

    ax.xaxis.grid(False)

    bin_label_locs = []
    all_rects = []
    outer_min_x_loc = 0
    middle_min_x_loc = 0
    inner_min_x_loc = 0

    circ_patches = []
    max_bin_height = 0

    all_sample_label_x_locs = []
    all_sample_percs = []

    for i, (up) in enumerate(uncertainty_types_list):
        uncertainty_type = up[0]
        for hash_idx, model_type in enumerate(models):
            inbetween_locs = []

            average_samples_per_bin = []
            for j in range(num_bins):
                if j == 0:
                    if hash_idx == 1:
                        circ11 = patches.Patch(
                            facecolor=cmaps[i],
                            label=model_type + " " + uncertainty_type,
                            hatch=hatch_type,
                            edgecolor="black",
                        )
                    else:
                        circ11 = patches.Patch(facecolor=cmaps[i], label=model_type + " " + uncertainty_type)
                    circ_patches.append(circ11)

                dict_key = [
                    x for x in list(landmark_uncert_dicts.keys()) if (model_type in x) and (uncertainty_type in x)
                ][0]
                model_data = landmark_uncert_dicts[dict_key]
                all_b_data = model_data[j]

                # if j == num_bins-1:
                logger.info(
                    "Bin %s (len=%s), model: %s, uncertainty: %s, and mean error: %s +/- %s",
                    j,
                    len(all_b_data),
                    model_type,
                    uncertainty_type,
                    np.round(np.mean(all_b_data), 2),
                    np.round(np.std(all_b_data), 2),
                )
                # print("and all data: ", all_b_data)

                orders.append(model_type + uncertainty_type)

                width = 0.25

                x_loc = [(outer_min_x_loc + inner_min_x_loc + middle_min_x_loc)]
                inbetween_locs.append(x_loc[0])

                # Turn data to percentages
                if turn_to_percent:
                    displayed_data = [(x) * 100 for x in all_b_data]
                else:
                    displayed_data = all_b_data
                rect = ax.boxplot(
                    displayed_data, positions=x_loc, sym="", widths=width, showmeans=True, patch_artist=True
                )

                if show_individual_dots:
                    # Add some random "jitter" to the x-axis
                    x = np.random.normal(x_loc, 0.01, size=len(displayed_data))
                    ax.plot(
                        x,
                        displayed_data,
                        color=cmaps[len(uncertainty_types_list)],
                        marker=".",
                        linestyle="None",
                        alpha=0.75,
                    )

                # Set colour, pattern, median line and mean marker.
                for r in rect["boxes"]:
                    r.set(color="black", linewidth=1)
                    r.set(facecolor=cmaps[i])

                    if hash_idx == 1:
                        r.set_hatch(hatch_type)
                for median in rect["medians"]:
                    median.set(color="crimson", linewidth=3)

                for mean in rect["means"]:
                    mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=10)

                max_bin_height = max(max(rect["caps"][-1].get_ydata()), max_bin_height)

                """If we are showing sample info, keep track of it and display after on top of biggest whisker."""
                if show_sample_info != "None":
                    flattened_model_data = [x for xss in model_data for x in xss]
                    percent_size = np.round(len(all_b_data) / len(flattened_model_data) * 100, 1)
                    average_samples_per_bin.append(percent_size)

                    if show_sample_info == "All":
                        """This adds the number of samples on top of the top whisker"""
                        (x_l, y), (x_r, _) = rect["caps"][-1].get_xydata()
                        x_line_center = (x_l + x_r) / 2
                        all_sample_label_x_locs.append(x_line_center)
                        all_sample_percs.append(percent_size)
                all_rects.append(rect)

                inner_min_x_loc += 0.1 + width

            """ Keep track of average sample infos. Plot at the END so we know what the max height for all Qs are."""
            if show_sample_info == "Average":
                middle_x = np.mean(inbetween_locs)
                mean_perc = np.round(np.mean(average_samples_per_bin), 1)
                std_perc = np.round(np.std(average_samples_per_bin), 1)
                all_sample_label_x_locs.append(middle_x)
                all_sample_percs.append([mean_perc, std_perc])

            bin_label_locs = bin_label_locs + inbetween_locs

            # IF lots of bins we must make the gap between plots bigger to prevent overlapping x-tick labels.
            if num_bins > 9:
                middle_min_x_loc += 0.25
            else:
                middle_min_x_loc += 0.12

        outer_min_x_loc += 0.24

    # Show the average samples on top of boxplots, aligned. if lots of bins we can lower the height.
    if show_sample_info != "None":
        if num_bins > 5:
            max_bin_height = max_bin_height * 0.8
        else:
            max_bin_height += 0.5
        for idx_text, perc_info in enumerate(all_sample_percs):
            if show_sample_info == "Average":
                ax.text(
                    all_sample_label_x_locs[idx_text],
                    max_bin_height,  # Position
                    r"$\bf{PSB}$" + ": \n" + r"${} \pm$".format(perc_info[0]) + "\n" + r"${}$".format(perc_info[1]),
                    verticalalignment="bottom",  # Centered bottom with line
                    horizontalalignment="center",  # Centered with horizontal line
                    fontsize=25,
                )
            elif show_sample_info == "All":
                if idx_text % 2 == 0:
                    label_height = max_bin_height + 1.5
                else:
                    label_height = max_bin_height
                ax.text(
                    all_sample_label_x_locs[idx_text],
                    label_height,  # Position
                    str(perc_info) + "%",
                    horizontalalignment="center",  # Centered with horizontal line
                    fontsize=15,
                )

    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.set_xticks(bin_label_locs)

    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    # plt.axis("tight")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if num_bins <= 5:
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis_labels[:-1] * (len(uncertainty_types_list) * 2)))
    # If too many bins, only show the first and last or it will appear too squished, indicate direction with arrow.
    elif num_bins < 15:
        number_blanks_0 = ["" for x in range(math.floor((num_bins - 3) / 2))]
        number_blanks_1 = ["" for x in range(num_bins - 3 - len(number_blanks_0))]
        new_labels = [x_axis_labels[0]] + number_blanks_0 + [r"$\rightarrow$"] + number_blanks_1 + [x_axis_labels[-1]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))
    # if more than 15 bins, we must move the first and last labels inwards to prevent overlap.
    else:
        number_blanks_0 = ["" for x in range(math.floor((num_bins - 5) / 2))]
        number_blanks_1 = ["" for x in range(num_bins - 5 - len(number_blanks_0))]
        new_labels = (
            [""]
            + [x_axis_labels[0]]
            + number_blanks_0
            + [r"$\rightarrow$"]
            + number_blanks_1
            + [x_axis_labels[-1]]
            + [""]
        )
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))

    if to_log:
        # ax.set_yscale("log",base=2)
        ax.set_yscale("symlog", base=2)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-0.1, y_lim)  # set the x ticks in aesthitically pleasing place

    else:
        ax.set_ylim((-0.1, y_lim))  # set the x ticks in aesthitically pleasing place

    # Add more to legend, add the mean symbol and median symbol.
    red_triangle_mean = mlines.Line2D(
        [], [], color="crimson", marker="^", markeredgecolor="black", linestyle="None", markersize=10, label="Mean"
    )
    circ_patches.append(red_triangle_mean)

    red_line_median = mlines.Line2D(
        [], [], color="crimson", marker="", markeredgecolor="black", markersize=10, label="Median"
    )
    circ_patches.append(red_line_median)

    if show_sample_info == "Average":
        circ_patches.append(patches.Patch(color="none", label=r"$\bf{PSB}$" + r": % Samples per Bin"))

    num_cols_legend = math.ceil(len(circ_patches) / 2)
    # ax.legend(handles=circ_patches, loc=9, fontsize=15, ncol=num_cols_legend, columnspacing=3)
    ax.legend(
        handles=circ_patches,
        fontsize=20,
        ncol=num_cols_legend,
        columnspacing=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fancybox=True,
        shadow=False,
    )
    #       ncol=3, fancybox=True, shadow=True)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=3, fancybox=True, shadow=True)

    # plt.autoscale()
    if save_path is not None:
        plt.gcf().set_size_inches(12.0, 7.5)
        # plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 8.0)
        # plt.gcf().set_size_inches(12.0, 7.5)
        plt.tight_layout()
        plt.show()
        plt.close()


def box_plot_comparing_q(
    landmark_uncert_dicts_list,
    uncertainty_type_tuple,
    model,
    x_axis_labels,
    x_label,
    y_label,
    num_bins_display,
    hatch_type,
    colour,
    show_sample_info="None",
    save_path=None,
    y_lim=120,
    turn_to_percent=True,
    to_log=False,
    show_individual_dots=True,
):
    """
    Creates a box plot of data, using Q (# Bins) on the x-axis. Only compares 1 model & 1 uncertainty type using Q on the x-axis.

    Args:
        landmark_uncert_dicts_list ([Dict]): List of Dict of pandas dataframe for the data to dsiplay, 1 for each value for Q.
        uncertainty_type ([str]): list describing the single uncertainty/error type to display,
        model (str): the model we are comparing over our values of Q
        x_axis_labels (list): list of strings for the x-axis labels, one for each bin,
        x_label (str): x axis label,
        y_label (int): y axis label,
        hatch_type (str): hatch type for the box plot,
        colour (str): colour for the box plot,
        num_bins ([int]): List of values of Q (#bins) we are comparing on our x-axis.
        save_path (str):path to save plot to. If None, displays on screen (default=None),
        y_lim (int): y axis limit of graph (default=120),


    """

    plt.style.use("fivethirtyeight")

    orders = []
    ax = plt.gca()

    # fig.set_size_inches(24, 10)

    ax.xaxis.grid(False)

    bin_label_locs = []
    all_rects = []
    outer_min_x_loc = 0
    inner_min_x_loc = 0
    middle_min_x_loc = 0

    circ_patches = []

    uncertainty_type = uncertainty_type_tuple[0][0]
    model_type = model[0]

    # Set legend
    circ11 = patches.Patch(
        hatch=hatch_type, facecolor=colour, label=model_type + " " + uncertainty_type, edgecolor="black",
    )
    circ_patches.append(circ11)

    max_bin_height = 0
    all_sample_label_x_locs = []
    all_sample_percs = []

    for idx, q_value in enumerate(x_axis_labels):
        inbetween_locs = []
        landmark_uncert_dicts = landmark_uncert_dicts_list[idx]

        # Get key for the model and uncetainty type for data
        dict_key = [x for x in list(landmark_uncert_dicts.keys()) if (model_type in x) and (uncertainty_type in x)][0]
        model_data = landmark_uncert_dicts[dict_key]
        average_samples_per_bin = []
        # Loop through each bin and display the data
        for j in range(len(model_data)):
            all_b_data = model_data[j]

            orders.append(model_type + uncertainty_type)

            width = 0.2 * (4 / 5) ** idx

            x_loc = [(outer_min_x_loc + inner_min_x_loc + middle_min_x_loc)]
            inbetween_locs.append(x_loc[0])

            # Turn data to percentages
            if turn_to_percent:
                displayed_data = [(x) * 100 for x in all_b_data]
            else:
                displayed_data = all_b_data
            rect = ax.boxplot(displayed_data, positions=x_loc, sym="", widths=width, showmeans=True, patch_artist=True)

            max_bin_height = max(max(rect["caps"][-1].get_ydata()), max_bin_height)

            if show_individual_dots:
                # Add some random "jitter" to the x-axis
                x = np.random.normal(x_loc, 0.01, size=len(displayed_data))
                ax.plot(x, displayed_data, color="crimson", marker=".", linestyle="None", alpha=0.2)

            # Set colour, pattern, median line and mean marker.
            for r in rect["boxes"]:
                r.set(color="black", linewidth=1)
                r.set(facecolor=colour)
                r.set_hatch(hatch_type)
            for median in rect["medians"]:
                median.set(color="crimson", linewidth=3)

            for mean in rect["means"]:
                mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=10)

            """If we are showing sample info, keep track of it and display after on top of biggest whisker."""
            if show_sample_info != "None":
                flattened_model_data = [x for xss in model_data for x in xss]
                percent_size = np.round(len(all_b_data) / len(flattened_model_data) * 100, 1)
                average_samples_per_bin.append(percent_size)

                if show_sample_info == "All":
                    """This adds the number of samples on top of the top whisker"""
                    (x_l, y), (x_r, _) = rect["caps"][-1].get_xydata()
                    x_line_center = (x_l + x_r) / 2
                    all_sample_label_x_locs.append([x_line_center, y + 0.5])
                    all_sample_percs.append(percent_size)

            all_rects.append(rect)
            inner_min_x_loc += 0.02 + width

        outer_min_x_loc += 0.2
        bin_label_locs.append(np.mean(inbetween_locs))

        """ Keep track of average sample infos. Plot at the END so we know what the max height for all Qs are."""
        if show_sample_info == "Average":
            middle_x = np.mean(inbetween_locs)
            mean_perc = np.round(np.mean(average_samples_per_bin), 1)
            std_perc = np.round(np.std(average_samples_per_bin), 1)
            all_sample_label_x_locs.append(middle_x)
            all_sample_percs.append([mean_perc, std_perc])

    # if show_sample_info == "Average":
    #     max_bin_height += 0.5
    #     for idx_text, perc_info in enumerate(all_sample_percs):
    #         ax.text(all_sample_label_x_locs[idx_text], max_bin_height, # Position
    #             r"$\bf{ASB}$"  +": \n" + r"${} \pm$".format(perc_info[0]) + "\n" + r"${}$".format(perc_info[1]) + "%",
    #             verticalalignment='top', # Centered bottom with line
    #             horizontalalignment='center', # Centered with horizontal line
    #             fontsize=18
    #         )

    # Show the average samples on top of boxplots, aligned. if lots of bins we can lower the height.
    if show_sample_info != "None":
        for idx_text, perc_info in enumerate(all_sample_percs):
            if show_sample_info == "Average":
                ax.text(
                    all_sample_label_x_locs[idx_text],
                    max_bin_height * 0.8,  # Position
                    r"$\bf{PSB}$" + ": \n" + r"${} \pm$".format(perc_info[0]) + "\n" + r"${}$".format(perc_info[1]),
                    verticalalignment="bottom",  # Centered bottom with line
                    horizontalalignment="center",  # Centered with horizontal line
                    fontsize=25,
                )
            elif show_sample_info == "All":
                if idx_text % 2 == 0:
                    label_height = max_bin_height + 3
                else:
                    label_height = max_bin_height + 1
                ax.text(
                    all_sample_label_x_locs[idx_text][0],
                    label_height,  # Position
                    r"$\bf{PSB}$" + ": \n" + str(perc_info) + "%",
                    verticalalignment="bottom",  # Centered bottom with line
                    horizontalalignment="center",  # Centered with horizontal line
                    fontsize=25,
                )

    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.set_xticks(bin_label_locs)

    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis_labels))

    if to_log:
        # ax.set_yscale("log",base=2)
        ax.set_yscale("symlog", base=2)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-0.1, y_lim)  # set the x ticks in aesthitically pleasing place

    else:
        ax.set_ylim((-0.1, y_lim))  # set the x ticks in aesthitically pleasing place

    # Add more to legend, add the mean symbol and median symbol.

    red_triangle_mean = mlines.Line2D(
        [], [], color="crimson", marker="^", markeredgecolor="black", linestyle="None", markersize=10, label="Mean"
    )
    circ_patches.append(red_triangle_mean)

    red_line_median = mlines.Line2D(
        [], [], color="crimson", marker="", markeredgecolor="black", markersize=10, label="Median"
    )
    circ_patches.append(red_line_median)

    if show_sample_info == "Average":
        circ_patches.append(patches.Patch(color="none", label=r"$\bf{PSB}$" + r": % Samples per Bin"))
    elif show_sample_info == "All":
        circ_patches.append(patches.Patch(color="none", label=r"$\bf{PSB}$" + r": % Samples per Bin"))

    ax.legend(handles=circ_patches, loc=9, fontsize=20, ncol=4, columnspacing=6)
    # plt.autoscale()
    if save_path is not None:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_cumulative(
    cmaps,
    data_struct,
    models,
    uncertainty_types,
    bins,
    title,
    compare_to_all=False,
    save_path=None,
    pixel_to_mm_scale=1,
):
    """
    Plots cumulative errors.

    Args:
        cmaps (list): list of colours for matplotlib,
        data_struct (Dict): Dict of pandas dataframe for the data to display,
        models (list): the models we want to compare, keys in landmark_uncert_dicts,
        uncertainty_types ([list]): list of lists describing the different uncert combinations to test,
        bins (list): List of bins to show error form,
        compare_to_all (bool): Whether to compare the given subset of bins to all the data (default=False)
        save_path (str):path to save plot to. If None, displays on screen (default=None),
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
    # ax.set_xlim(0, 30)
    line_styles = [":", "-", "dotted", "-."]
    for i, (up) in enumerate(uncertainty_types):
        uncertainty = up[0]
        colour = cmaps[i]
        for hash_idx, model_type in enumerate(models):
            line = line_styles[hash_idx]

            # Filter only the bins selected
            dataframe = data_struct[model_type]
            model_un_errors = (
                dataframe[dataframe[uncertainty + " Uncertainty bins"].isin(bins)][uncertainty + " Error"].values
                * pixel_to_mm_scale
            )

            p = 100 * np.arange(len(model_un_errors)) / (len(model_un_errors) - 1)

            sorted_errors = np.sort(model_un_errors)

            ax.plot(
                sorted_errors,
                p,
                label=model_type + " " + uncertainty,
                color=colour,
                linestyle=line,
                dash_capstyle="round",
            )

            if compare_to_all:
                dataframe = data_struct[model_type]
                model_un_errors = dataframe[uncertainty + " Error"].values * pixel_to_mm_scale

                p = 100 * np.arange(len(model_un_errors)) / (len(model_un_errors) - 1)

                sorted_errors = np.sort(model_un_errors)
                line = line_styles[len(models) + hash_idx]
                ax.plot(
                    sorted_errors,
                    p,
                    label=model_type + " " + uncertainty,
                    color=colour,
                    linestyle=line,
                    dash_capstyle="round",
                )

    handles, labels = ax.get_legend_handles_labels()
    # ax2.legend(loc=2})
    ax.legend(handles, labels, prop={"size": 10})
    plt.axvline(x=5, color=cmaps[3])

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    plt.xticks([1, 2, 3, 4, 5, 10, 20, 30])

    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0)
        plt.show()
        plt.close()


def generate_figures_individual_bin_comparison(data, display_settings):
    logger = logging.getLogger("qbin")
    [
        uncertainty_error_pairs,
        models_to_compare,
        dataset,
        landmarks,
        num_bins,
        cmaps,
        save_folder,
        save_file_preamble,
        cfg,
        show_individual_landmark_plots,
        interpret,
        num_folds,
        ind_landmarks_to_show,
        pixel_to_mm_scale,
    ] = data

    # If combining the middle bins we just have the 2 edge bins, and the combined middle ones.

    # saved_bins_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset, "res_predicted_bins")
    bins_all_lms, bins_lms_sep, bounds_all_lms, bounds_lms_sep = get_data_struct(
        models_to_compare, landmarks, save_folder, dataset
    )

    # Get mean errors bin-wise, get all errors concatenated together bin-wise, and seperate by landmark.
    all_error_data_dict = get_mean_errors(
        bins_all_lms,
        uncertainty_error_pairs,
        num_bins,
        landmarks,
        num_folds=num_folds,
        pixel_to_mm_scale=pixel_to_mm_scale,
        combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
    )
    all_error_data = all_error_data_dict["all mean error bins nosep"]

    all_bins_concat_lms_nosep_error = all_error_data_dict["all error concat bins lms nosep"]  # shape is [num bins]

    all_bins_concat_lms_sep_all_error = all_error_data_dict[
        "all error concat bins lms sep all"
    ]  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

    # Get correlation coefficients for all bins

    # Get jaccard
    all_jaccard_data_dict = evaluate_jaccard(
        bins_all_lms,
        uncertainty_error_pairs,
        num_bins,
        landmarks,
        num_folds=num_folds,
        combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
    )
    all_jaccard_data = all_jaccard_data_dict["Jaccard All"]
    all_recall_data = all_jaccard_data_dict["Recall All"]
    all_precision_data = all_jaccard_data_dict["Precision All"]

    all_bins_concat_lms_sep_all_jacc = all_jaccard_data_dict[
        "all jacc concat bins lms sep all"
    ]  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

    bound_return_dict = evaluate_bounds(
        bounds_all_lms,
        bins_all_lms,
        uncertainty_error_pairs,
        num_bins,
        landmarks,
        num_folds,
        combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
    )

    all_bound_data = bound_return_dict["Error Bounds All"]

    all_bins_concat_lms_sep_all_errorbound = bound_return_dict[
        "all errorbound concat bins lms sep all"
    ]  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

    # x = generate_summary_df(all_error_data_dict, ))
    # print("UEP ", uncertainty_error_pairs)
    generate_summary_df(
        all_error_data_dict,
        [["all mean error bins nosep", "All Landmarks"]],
        "Mean error",
        os.path.join(save_folder, "localisation_errors.xlsx"),
    )

    # exit()

    if interpret:
        save_location = None

        # If we have combined the middle bins, we are only displaying 3 bins (outer edges, and combined middle bins).
        if cfg.PIPELINE.COMBINE_MIDDLE_BINS:
            num_bins_display = 3
        else:
            num_bins_display = num_bins

        # Plot piecewise linear regression for error/uncertainty prediction.
        if display_settings["correlation"]:
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = save_folder
            _ = evaluate_correlations(
                bins_all_lms,
                uncertainty_error_pairs,
                cmaps,
                num_bins,
                cfg.DATASET.CONFIDENCE_INVERT,
                num_folds=num_folds,
                pixel_to_mm_scale=pixel_to_mm_scale,
                combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
                save_path=save_location,
                to_log=True,
            )

        # Plot cumulative error figure for all predictions
        if display_settings["cumulative_error"]:
            plot_cumulative(
                cmaps,
                bins_all_lms,
                models_to_compare,
                uncertainty_error_pairs,
                np.arange(num_bins),
                "Cumulative error for ALL predictions, dataset " + dataset,
                save_path=None,
                pixel_to_mm_scale=pixel_to_mm_scale,
            )
            # Plot cumulative error figure for B1 only predictions
            plot_cumulative(
                cmaps,
                bins_all_lms,
                models_to_compare,
                uncertainty_error_pairs,
                0,
                "Cumulative error for B1 predictions, dataset " + dataset,
                save_path=None,
                pixel_to_mm_scale=pixel_to_mm_scale,
            )

            # Plot cumulative error figure comparing B1 and ALL, for both models
            for model_type in models_to_compare:
                plot_cumulative(
                    cmaps,
                    bins_all_lms,
                    [model_type],
                    uncertainty_error_pairs,
                    0,
                    model_type + ". Cumulative error comparing ALL and B1, dataset " + dataset,
                    compare_to_all=True,
                    save_path=None,
                    pixel_to_mm_scale=pixel_to_mm_scale,
                )

        # Set x_axis labels for following plots.
        x_axis_labels = [r"$B_{{{}}}$".format(num_bins_display + 1 - (i + 1)) for i in range(num_bins_display + 1)]

        # get error bounds

        if display_settings["errors"]:
            # mean error concat for each bin
            logger.info("mean error concat all L")
            if cfg.OUTPUT.SAVE_FIGURES:
                if cfg.BOXPLOT.SAMPLES_AS_DOTS:
                    dotted_addition = "_dotted"
                else:
                    dotted_addition = "_undotted"
                save_location = os.path.join(save_folder, save_file_preamble + dotted_addition + "_error_all_lms.pdf")

            box_plot_per_model(
                cmaps,
                all_bins_concat_lms_nosep_error,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Localization Error (mm)",
                num_bins=num_bins_display,
                turn_to_percent=False,
                show_sample_info=cfg.BOXPLOT.SHOW_SAMPLE_INFO_MODE,
                show_individual_dots=cfg.BOXPLOT.SAMPLES_AS_DOTS,
                y_lim=cfg.BOXPLOT.ERROR_LIM,
                to_log=True,
                save_path=save_location,
            )

            if show_individual_landmark_plots:
                # plot the concatentated errors for each landmark seperately
                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_error):
                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + dotted_addition + "_error_lm_" + str(idx_l) + ".pdf"
                            )

                        logger.info("individual error for L", idx_l)
                        box_plot_per_model(
                            cmaps,
                            lm_data,
                            uncertainty_error_pairs,
                            models_to_compare,
                            x_axis_labels=x_axis_labels,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Error (mm)",
                            num_bins=num_bins_display,
                            turn_to_percent=False,
                            show_sample_info=cfg.BOXPLOT.SHOW_SAMPLE_INFO_MODE,
                            show_individual_dots=cfg.BOXPLOT.SAMPLES_AS_DOTS,
                            y_lim=cfg.BOXPLOT.ERROR_LIM,
                            to_log=True,
                            save_path=save_location,
                        )

            logger.info("Mean error")

            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(
                    save_folder, save_file_preamble + dotted_addition + "mean_error_folds_all_lms.pdf"
                )
            box_plot_per_model(
                cmaps,
                all_error_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Mean Error (mm)",
                num_bins=num_bins_display,
                turn_to_percent=False,
                y_lim=cfg.BOXPLOT.ERROR_LIM,
                to_log=True,
                save_path=save_location,
            )

        # Plot Error Bound Accuracy

        if display_settings["error_bounds"]:
            logger.info(" errorbound acc for all landmarks.")
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_errorbound_all_lms.pdf")

            box_plot(
                cmaps,
                all_bound_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Error Bound Accuracy (%)",
                num_bins=num_bins_display,
                save_path=save_location,
                y_lim=120,
            )

            if show_individual_landmark_plots:
                # plot the concatentated error bounds for each landmark seperately
                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_errorbound):
                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_errorbound_lm_" + str(idx_l) + ".pdf"
                            )

                        logger.info("individual errorbound acc for L", idx_l)
                        box_plot(
                            cmaps,
                            lm_data,
                            uncertainty_error_pairs,
                            models_to_compare,
                            x_axis_labels=x_axis_labels,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Error Bound Accuracy (%)",
                            num_bins=num_bins_display,
                            save_path=save_location,
                            y_lim=120,
                        )

        # Plot Jaccard Index
        if display_settings["jaccard"]:
            logger.info("Plot jaccard for all landmarks.")
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_jaccard_all_lms.pdf")
            box_plot(
                cmaps,
                all_jaccard_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Jaccard Index (%)",
                num_bins=num_bins_display,
                y_lim=70,
                show_sample_info="None",
                save_path=save_location,
            )

            # mean recall for each bin
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_recall_jaccard_all_lms.pdf")
            box_plot(
                cmaps,
                all_recall_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Recall",
                num_bins=num_bins_display,
                turn_to_percent=True,
                y_lim=120,
                save_path=save_location,
            )

            # mean precision for each bin
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_precision_jaccard_all_lms.pdf")
            box_plot(
                cmaps,
                all_precision_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Ground Truth Bins Precision",
                num_bins=num_bins_display,
                turn_to_percent=True,
                y_lim=120,
                save_path=save_location,
            )

            if show_individual_landmark_plots:
                # plot the jaccard index for each landmark seperately

                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_jacc):
                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "jaccard_lm_" + str(idx_l) + ".pdf"
                            )

                        logger.info("individual jaccard for L", idx_l)
                        box_plot(
                            cmaps,
                            lm_data,
                            uncertainty_error_pairs,
                            models_to_compare,
                            x_axis_labels=x_axis_labels,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Jaccard Index (%)",
                            num_bins=num_bins_display,
                            y_lim=70,
                            save_path=save_location,
                        )


def generate_figures_comparing_bins(data, display_settings):
    [
        uncertainty_error_pair,
        model,
        dataset,
        landmarks,
        all_values_q,
        cmaps,
        all_fitted_save_paths,
        save_folder,
        save_file_preamble,
        cfg,
        show_individual_landmark_plots,
        interpret,
        num_folds,
        ind_landmarks_to_show,
        pixel_to_mm_scale,
    ] = data

    logger = logging.getLogger("qbin")

    hatch = display_settings["hatch"]
    colour = display_settings["colour"]

    # increse dimension of these for compatibility with future methods
    model = [model]
    uncertainty_error_pair = [uncertainty_error_pair]

    # If combining the middle bins we just have the 2 edge bins, and the combined middle ones.

    all_error_data = []
    all_error_lm_sep = []
    all_bins_concat_lms_nosep_error = []
    all_bins_concat_lms_sep_foldwise_error = []
    all_bins_concat_lms_sep_all_error = []
    all_jaccard_data = []
    all_recall_data = []
    all_precision_data = []
    all_bins_concat_lms_sep_foldwise_jacc = []
    all_bins_concat_lms_sep_all_jacc = []
    all_bound_data = []
    all_bins_concat_lms_sep_foldwise_errorbound = []
    all_bins_concat_lms_sep_all_errorbound = []

    for idx, num_bins in enumerate(all_values_q):
        saved_bins_path_pre = all_fitted_save_paths[idx]

        bins_all_lms, bins_lms_sep, bounds_all_lms, bounds_lms_sep = get_data_struct(
            model, landmarks, saved_bins_path_pre, dataset
        )

        # Get mean errors bin-wise, get all errors concatenated together bin-wise, and seperate by landmark.
        all_error_data_dict = get_mean_errors(
            bins_all_lms,
            uncertainty_error_pair,
            num_bins,
            landmarks,
            num_folds=num_folds,
            pixel_to_mm_scale=pixel_to_mm_scale,
            combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
        )
        all_error_data.append(all_error_data_dict["all mean error bins nosep"])
        all_error_lm_sep.append(all_error_data_dict["all mean error bins lms sep"])

        all_bins_concat_lms_nosep_error.append(
            all_error_data_dict["all error concat bins lms nosep"]
        )  # shape is [num bins]
        all_bins_concat_lms_sep_foldwise_error.append(
            all_error_data_dict["all error concat bins lms sep foldwise"]
        )  # shape is [num lms][num bins]
        all_bins_concat_lms_sep_all_error.append(
            all_error_data_dict["all error concat bins lms sep all"]
        )  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

        all_jaccard_data_dict = evaluate_jaccard(
            bins_all_lms,
            uncertainty_error_pair,
            num_bins,
            landmarks,
            num_folds=num_folds,
            combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
        )
        all_jaccard_data.append(all_jaccard_data_dict["Jaccard All"])
        all_recall_data.append(all_jaccard_data_dict["Recall All"])
        all_precision_data.append(all_jaccard_data_dict["Precision All"])
        all_bins_concat_lms_sep_foldwise_jacc.append(
            all_jaccard_data_dict["all jacc concat bins lms sep foldwise"]
        )  # shape is [num lms][num bins]
        all_bins_concat_lms_sep_all_jacc.append(
            all_jaccard_data_dict["all jacc concat bins lms sep all"]
        )  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

        bound_return_dict = evaluate_bounds(
            bounds_all_lms,
            bins_all_lms,
            uncertainty_error_pair,
            num_bins,
            landmarks,
            num_folds,
            combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS,
        )

        all_bound_data.append(bound_return_dict["Error Bounds All"])
        all_bins_concat_lms_sep_foldwise_errorbound.append(
            bound_return_dict["all errorbound concat bins lms sep foldwise"]
        )  # shape is [num lms][num bins]
        all_bins_concat_lms_sep_all_errorbound.append(
            bound_return_dict["all errorbound concat bins lms sep all"]
        )  # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list

    if interpret:
        # If we have combined the middle bins, we are only displaying 3 bins (outer edges, and combined middle bins).
        if cfg.PIPELINE.COMBINE_MIDDLE_BINS:
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
            if cfg.OUTPUT.SAVE_FIGURES:
                if cfg.BOXPLOT.SAMPLES_AS_DOTS:
                    dotted_addition = "_dotted"
                else:
                    dotted_addition = "_undotted"
                save_location = os.path.join(save_folder, save_file_preamble + dotted_addition + "_error_all_lms.pdf")

            box_plot_comparing_q(
                all_bins_concat_lms_nosep_error,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Localization Error (mm)",
                num_bins_display=num_bins_display,
                turn_to_percent=False,
                show_sample_info=cfg.BOXPLOT.SHOW_SAMPLE_INFO_MODE,
                show_individual_dots=cfg.BOXPLOT.SAMPLES_AS_DOTS,
                y_lim=cfg.BOXPLOT.ERROR_LIM,
                to_log=True,
                save_path=save_location,
            )

            if show_individual_landmark_plots:
                # plot the concatentated errors for each landmark seperately. Must transpose the iteration.
                for lm in landmarks:
                    lm_data = [x[lm] for x in all_bins_concat_lms_sep_all_error]

                    if lm in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + dotted_addition + "_error_lm_" + str(lm) + ".pdf"
                            )

                        logger.info("individual error for L", lm)
                        box_plot_comparing_q(
                            lm_data,
                            uncertainty_error_pair,
                            model,
                            hatch_type=hatch,
                            colour=colour,
                            x_axis_labels=x_axis_labels,
                            x_label="Q (# Bins)",
                            y_label="Localization Error (mm)",
                            num_bins_display=num_bins_display,
                            turn_to_percent=False,
                            show_sample_info=cfg.BOXPLOT.SHOW_SAMPLE_INFO_MODE,
                            show_individual_dots=cfg.BOXPLOT.SAMPLES_AS_DOTS,
                            y_lim=cfg.BOXPLOT.ERROR_LIM,
                            to_log=True,
                            save_path=save_location,
                        )

            logger.info("mean error")

            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(
                    save_folder, save_file_preamble + dotted_addition + "mean_error_folds_all_lms.pdf"
                )
            box_plot_comparing_q(
                all_error_data,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Mean Error (mm)",
                num_bins_display=num_bins_display,
                turn_to_percent=False,
                show_sample_info="None",
                show_individual_dots=False,
                y_lim=cfg.BOXPLOT.ERROR_LIM,
                to_log=True,
                save_path=save_location,
            )

        # Plot Error Bound Accuracy

        if display_settings["error_bounds"]:
            logger.info(" errorbound acc for all landmarks.")
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_errorbound_all_lms.pdf")

            box_plot_comparing_q(
                all_bound_data,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Error Bound Accuracy (%)",
                num_bins_display=num_bins_display,
                turn_to_percent=True,
                show_sample_info="None",
                show_individual_dots=False,
                y_lim=100,
                save_path=save_location,
            )

            if show_individual_landmark_plots:
                # plot the concatentated errors for each landmark seperately. Must transpose the iteration.
                for lm in landmarks:
                    lm_data = [x[lm] for x in all_bins_concat_lms_sep_all_errorbound]

                    if lm in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "_errorbound_lm_" + str(lm) + ".pdf"
                            )

                        logger.info("individual errorbound acc for L", lm)
                        box_plot_comparing_q(
                            lm_data,
                            uncertainty_error_pair,
                            model,
                            hatch_type=hatch,
                            colour=colour,
                            x_axis_labels=x_axis_labels,
                            x_label="Q (# Bins)",
                            y_label="Error Bound Accuracy (%)",
                            num_bins_display=num_bins_display,
                            turn_to_percent=True,
                            show_individual_dots=False,
                            y_lim=100,
                            save_path=save_location,
                        )

        # Plot Jaccard Index
        if display_settings["jaccard"]:
            logger.info("Plot jaccard for all landmarks.")
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_jaccard_all_lms.pdf")

            box_plot_comparing_q(
                all_jaccard_data,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Jaccard Index (%)",
                num_bins_display=num_bins_display,
                turn_to_percent=True,
                show_individual_dots=False,
                y_lim=70,
                save_path=save_location,
            )

            # mean recall for each bin
            logger.info("Plot recall for all landmarks.")

            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_recall_jaccard_all_lms.pdf")
            box_plot_comparing_q(
                all_recall_data,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bin Recall (%)",
                num_bins_display=num_bins_display,
                turn_to_percent=True,
                show_individual_dots=False,
                y_lim=120,
                save_path=save_location,
            )

            # mean precision for each bin
            logger.info("Plot precision for all landmarks.")

            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble + "_precision_jaccard_all_lms.pdf")
            box_plot_comparing_q(
                all_precision_data,
                uncertainty_error_pair,
                model,
                hatch_type=hatch,
                colour=colour,
                x_axis_labels=x_axis_labels,
                x_label="Q (# Bins)",
                y_label="Ground Truth Bin Precision (%)",
                num_bins_display=num_bins_display,
                turn_to_percent=True,
                show_individual_dots=False,
                y_lim=120,
                save_path=save_location,
            )

            if show_individual_landmark_plots:
                # plot the concatentated errors for each landmark seperately. Must transpose the iteration.
                for lm in landmarks:
                    lm_data = [x[lm] for x in all_bins_concat_lms_sep_all_jacc]

                    if lm in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:
                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(
                                save_folder, save_file_preamble + "jaccard_lm_" + str(lm) + ".pdf"
                            )

                        logger.info("individual jaccard for L", lm)
                        box_plot_comparing_q(
                            lm_data,
                            uncertainty_error_pair,
                            model,
                            hatch_type=hatch,
                            colour=colour,
                            x_axis_labels=x_axis_labels,
                            x_label="Q (# Bins)",
                            y_label="Jaccard Index (%)",
                            num_bins_display=num_bins_display,
                            turn_to_percent=True,
                            show_individual_dots=False,
                            y_lim=70,
                            save_path=save_location,
                        )

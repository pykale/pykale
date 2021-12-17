# =============================================================================
# Author: Lawrence Schobs, laschobs1@sheffield.ac.uk
# =============================================================================

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.isotonic import IsotonicRegression


def quantile_binning_and_est_errors(errors, uncertainties, num_bins, type="quantile", acceptable_thresh=5 * 0.9375):
    """Calculate quantile thresholds, and isotonically regress errors and uncertainties and get estimated error bounds.

    Args:
        errors (list): list of errors,
        uncertainties (list): list of uncertainties,
        num_bins (int): Number of quantile bins,

        type (str): what type of thresholds to calculate. quantile recommended. (default='quantile),
        acceptable_thresh (float):acceptable error threshold. only relevent if type="error-wise".


    Returns:
        [list,list]: list of quantile thresholds and estimated error bounds.
    """
    valid_types = {"quantile", "error-wise"}
    if type not in valid_types:
        raise ValueError("results: type must be one of %r. " % valid_types)
    

    # Isotonically regress line
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    _ = ir.fit_transform(uncertainties, errors)

    uncert_boundaries = []
    estimated_errors = []

    if type == "quantile":
        quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)

        for q in range(len(quantiles)):
            q_conf_higher = [np.quantile(uncertainties, quantiles[q])]
            q_error_higher = ir.predict(q_conf_higher)

            estimated_errors.append(q_error_higher[0])
            uncert_boundaries.append(q_conf_higher)

    elif type == "error_wise":
        quantiles = np.arange(num_bins - 1)
        estimated_errors = [[(acceptable_thresh * x)] for x in quantiles]

        uncert_boundaries = [(ir.predict(x)).tolist() for x in estimated_errors]

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
    save_path=None,
    y_lim=120,
):
    """Creates a box plot of data.

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
    fig = plt.figure()
    ax = plt.gca()

    fig.set_size_inches(24, 10)

    ax.xaxis.grid(False)

    bin_label_locs = []
    all_rects = []
    outer_min_x_loc = 0
    middle_min_x_loc = 0
    inner_min_x_loc = 0

    max_error = 0
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

                width = 0.08
                x_loc = [(outer_min_x_loc + inner_min_x_loc + middle_min_x_loc)]
                inbetween_locs.append(x_loc[0])

                # Turn data to percentages
                percent_data = [(x) * 100 for x in all_b_data]
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
                    mean.set(markerfacecolor="crimson", markeredgecolor="black", markersize=15)

                for whisker in rect["whiskers"]:
                    max_error = max(max(whisker.get_ydata()), max_error)

                all_rects.append(rect)

                inner_min_x_loc += 0.0075 + width
            bin_label_locs.append(np.mean(inbetween_locs))
            middle_min_x_loc += 0.02
        outer_min_x_loc += 0.12

    ax.set_xlabel(x_label, fontsize=35)
    ax.set_ylabel(y_label, fontsize=35)
    ax.set_xticks(bin_label_locs)

    plt.subplots_adjust(bottom=0.15)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=25)

    ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis_labels[:-1] * (len(uncertainty_types_list) * 2)))

    ax.set_ylim((-2, y_lim))

    ax.legend(handles=circ_patches, loc=9, fontsize=25, ncol=3, columnspacing=6)

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_cumulative(cmaps, data_struct, models, uncertainty_types, bins, save_path=None):
    """Plots cumulative errors,

    Args:
        cmaps (list): list of colours for matplotlib,
        data_struct (Dict): Dict of pandas dataframe for the data to display,
        models (list): the models we want to compare, keys in landmark_uncert_dicts,
        uncertainty_types ([list]): list of lists describing the different uncert combinations to test,
        bins (list): List of bins to show error form
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

    ax.set_xscale("log")
    ax.set_xlim(0, 30)
    line_styles = ["-", ":", "dotted", "-."]
    for i, (up) in enumerate(uncertainty_types):
        uncertainty = up[0]
        colour = cmaps[i]
        for hash_idx, model_type in enumerate(models):
            line = line_styles[hash_idx]

            # Filter inly the bins selected
            dataframe = data_struct[model_type]
            model_un_errors = dataframe[dataframe[uncertainty + " Uncertainty bins"].isin(bins)][
                uncertainty + " Error"
            ].values

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

    handles, labels = ax.get_legend_handles_labels()
    # ax2.legend(loc=2})
    ax.legend(handles, labels, prop={"size": 10})
    plt.axvline(x=5, color=cmaps[3])

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.xticks([1, 2, 3, 4, 5, 10, 20, 30])

    ax.xaxis.label.set_color("black")  # setting up X-axis label color to yellow
    ax.yaxis.label.set_color("black")  # setting up Y-axis label color to blue

    ax.tick_params(axis="x", colors="black")  # setting up X-axis tick color to red
    ax.tick_params(axis="y", colors="black")  # setting up Y-axis tick color to black

    if save_path is not None:
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0.2)
        plt.close()
    else:
        plt.show()
        plt.close()

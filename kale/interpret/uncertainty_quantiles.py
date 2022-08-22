# =============================================================================
# Author: Lawrence Schobs, laschobs1@sheffield.ac.uk
# =============================================================================

import os
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.isotonic import IsotonicRegression

import matplotlib.lines as mlines
from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard, get_mean_errors

from kale.prepdata.tabular_transform import get_data_struct


def quantile_binning_and_est_errors(errors, uncertainties, num_bins, type="quantile", acceptable_thresh=5, combine_middle_bins=False):
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

    #Estimate error bounds for each quantile bin
    if type == "quantile":
        quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)[:num_bins-1]
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


    #IF combine bins, we grab only the values for the two outer bins
    if combine_middle_bins:
        estimated_errors = [estimated_errors[0],  estimated_errors[-1]]
        uncert_boundaries = [uncert_boundaries[0],  uncert_boundaries[-1]]


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
    show_sample_info ="None",
    save_path=None,
    y_lim=120,
    turn_to_percent=True,
    to_log= False,
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

                # for whisker in rect["whiskers"]:
                #     max_error = max(max(whisker.get_ydata()), max_error)

                all_rects.append(rect)

                inner_min_x_loc += 0.0075 + width
            bin_label_locs.append(np.mean(inbetween_locs))
            middle_min_x_loc += 0.02

        if num_bins >10:
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
    #If too many bins, only show the first and last or it will appear too squished, indicate direction with arrow.
    elif num_bins <15 :
        number_blanks_0 = ["" for x in range(math.floor((num_bins-3)/2))]
        number_blanks_1 = ["" for x in range(num_bins-3 - len(number_blanks_0))]
        new_labels = [x_axis_labels[-0]] + number_blanks_0 + [r"$\rightarrow$"]  +number_blanks_1+ [x_axis_labels[-1]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))
    #if more than 15 bins, we must move the first and last labels inwards to prevent overlap.
    else:
        number_blanks_0 = ["" for x in range(math.floor((num_bins-5)/2))]
        number_blanks_1 = ["" for x in range(num_bins-5 - len(number_blanks_0))]
        new_labels = [""] +[ x_axis_labels[0]] + number_blanks_0 + [r"$\rightarrow$"]  +number_blanks_1+ [x_axis_labels[-1] ] + [""]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))

    ax.legend(handles=circ_patches, loc=9, fontsize=25, ncol=3, columnspacing=6)

    if to_log:
        # ax.set_yscale("log",base=2)
        ax.set_yscale('symlog', base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-2, y_lim)

    else:
        ax.set_ylim((-2, y_lim))
    
    #If using percent, doesnt make sense to show any y tick above 100
    if turn_to_percent and y_lim > 100:
        plt.yticks(np.arange(0, y_lim, 20))

    ax.legend(handles=circ_patches, loc=9, fontsize=15, ncol=3, columnspacing=6)

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
    show_sample_info = "None",
    save_path=None,
    y_lim=120,
    turn_to_percent=True,
    to_log= False,
    show_individual_dots = True,

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

    max_error = 0
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
                #     print("Bin %s (len=%s), model: %s, uncertainty: %s, and mean error: %s" % (j,len(all_b_data), model_type, uncertainty_type, np.mean(all_b_data)))
                #     print("and all data: ", all_b_data)

                orders.append(model_type + uncertainty_type)

                width = 0.08

               
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
                    ax.plot(x, displayed_data, color=cmaps[len(uncertainty_types_list)], marker='.', linestyle="None", alpha=0.2)




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

                # for whisker in rect["whiskers"]:
                #     max_error = max(max(whisker.get_ydata()), max_error)
                max_bin_height = max(max(rect["caps"][-1].get_ydata()), max_bin_height)

                """If we are showing sample info, keep track of it and display after on top of biggest whisker."""
                if show_sample_info != "None":
                    flattened_model_data = [x for xss in model_data for x in xss]
                    percent_size = np.round(len(all_b_data) / len(flattened_model_data)*100, 1)
                    average_samples_per_bin.append(percent_size)

                    if show_sample_info == "All":
                        """ This adds the number of samples on top of the top whisker"""
                        (x_l, y),(x_r, _) = rect['caps'][-1].get_xydata()
                        x_line_center = (x_l+x_r)/2
                        all_sample_label_x_locs.append(x_line_center)
                        all_sample_percs.append(percent_size)
                all_rects.append(rect)

                
                inner_min_x_loc += 0.02 + width


            """ Keep track of average sample infos. Plot at the END so we know what the max height for all Qs are."""
            if show_sample_info == "Average":
                middle_x = np.mean(inbetween_locs)
                mean_perc = np.round(np.mean(average_samples_per_bin),1)
                std_perc = np.round(np.std(average_samples_per_bin),1)
                all_sample_label_x_locs.append(middle_x)
                all_sample_percs.append([mean_perc,std_perc ])



            bin_label_locs = bin_label_locs  + inbetween_locs

            #IF lots of bins we must make the gap between plots bigger to prevent overlapping x-tick labels.
            if num_bins >9:
                middle_min_x_loc += 0.25
            else:
                middle_min_x_loc += 0.12

        outer_min_x_loc += 0.18

    #Show the average samples on top of boxplots, aligned. if lots of bins we can lower the height.
    if show_sample_info != "None":
        if num_bins > 5:
            max_bin_height = max_bin_height *0.8
        else:
            max_bin_height += 0.5
        for idx_text, perc_info in enumerate(all_sample_percs):
            if show_sample_info == "Average":

                ax.text(all_sample_label_x_locs[idx_text], max_bin_height, # Position
                    r"$\bf{ASB}$"  +": \n" + r"${} \pm$".format(perc_info[0]) + "\n" + r"${}$".format(perc_info[1]) + "%", 
                    verticalalignment='bottom', # Centered bottom with line 
                    horizontalalignment='center', # Centered with horizontal line 
                    fontsize=18
                )
            elif show_sample_info == "All":
                if idx_text % 2 == 0:
                    label_height = max_bin_height + 1.5
                else:
                    label_height   = max_bin_height 
                ax.text(all_sample_label_x_locs[idx_text], label_height, # Position
                    str(perc_info) + "%", 
                    horizontalalignment='center', # Centered with horizontal line 
                    fontsize=15,)



    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.set_xticks(bin_label_locs)

    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)


    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if num_bins <= 5:
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_axis_labels[:-1] * (len(uncertainty_types_list) * 2)))
    #If too many bins, only show the first and last or it will appear too squished, indicate direction with arrow.
    elif num_bins <15 :
        number_blanks_0 = ["" for x in range(math.floor((num_bins-3)/2))]
        number_blanks_1 = ["" for x in range(num_bins-3 - len(number_blanks_0))]
        new_labels = [x_axis_labels[0]] + number_blanks_0 + [r"$\rightarrow$"]  +number_blanks_1+ [x_axis_labels[-1]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))
    #if more than 15 bins, we must move the first and last labels inwards to prevent overlap.
    else:
        number_blanks_0 = ["" for x in range(math.floor((num_bins-5)/2))]
        number_blanks_1 = ["" for x in range(num_bins-5 - len(number_blanks_0))]
        new_labels = [""] +[ x_axis_labels[0]] + number_blanks_0 + [r"$\rightarrow$"]  +number_blanks_1+ [x_axis_labels[-1] ] + [""]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(new_labels * (len(uncertainty_types_list) * 2)))




    if to_log:
        # ax.set_yscale("log",base=2)
        ax.set_yscale('symlog', base=2)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-0.1, y_lim) # set the x ticks in aesthitically pleasing place

    else:
        ax.set_ylim((-0.1, y_lim)) # set the x ticks in aesthitically pleasing place

    #Add more to legend, add the mean symbol and median symbol.
    red_triangle_mean = mlines.Line2D([], [], color='crimson', marker='^', markeredgecolor="black", linestyle='None',
                          markersize=10, label='Mean')
    circ_patches.append(red_triangle_mean)

    red_line_median = mlines.Line2D([], [], color='crimson', marker='', markeredgecolor="black",
                          markersize=10, label='Median')
    circ_patches.append(red_line_median)
    

    if show_sample_info == "Average":
        circ_patches.append(patches.Patch(color='none', label= r"$\bf{ASB}$" + r": Av. Samples Per Bin"))

    num_cols_legend = math.ceil(len(circ_patches)/2)
    ax.legend(handles=circ_patches, loc=9, fontsize=15, ncol=num_cols_legend, columnspacing=3)
    # plt.autoscale()
    if save_path is not None:
        plt.gcf().set_size_inches(16.0, 10.0) 
        # plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        plt.gcf().set_size_inches(16.0, 10.0) 
        plt.show()
        plt.close()






def generate_figures_individual_bin_comparison(data, display_settings):
    
    [uncertainty_error_pairs, models_to_compare, dataset, landmarks,num_bins, cmaps,save_folder,save_file_preamble,
     cfg, show_individual_landmark_plots, interpret, num_folds, ind_landmarks_to_show, pixel_to_mm_scale] =  data



    #If combining the middle bins we just have the 2 edge bins, and the combined middle ones.

    # saved_bins_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset, "res_predicted_bins")
    bins_all_lms, bins_lms_sep, bounds_all_lms, bounds_lms_sep = get_data_struct(
        models_to_compare, landmarks, save_folder, dataset
    )


    #Get mean errors bin-wise, get all errors concatenated together bin-wise, and seperate by landmark.
    all_error_data_dict = get_mean_errors(bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, num_folds=num_folds, pixel_to_mm_scale=pixel_to_mm_scale, combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS)
    all_error_data =  all_error_data_dict["all mean error bins nosep"]
    all_error_lm_sep = all_error_data_dict["all mean error bins lms sep"] 

    all_bins_concat_lms_nosep_error = all_error_data_dict["all error concat bins lms nosep"] # shape is [num bins]
    all_bins_concat_lms_sep_foldwise_error = all_error_data_dict["all error concat bins lms sep foldwise"] # shape is [num lms][num bins]
    all_bins_concat_lms_sep_all_error = all_error_data_dict["all error concat bins lms sep all"] # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list
    
    #Get correlation coefficients for all bins
    # all_correlation_data_dict = evaluate_correlations(bins_all_lms, uncertainty_error_pairs, cmaps, num_bins, landmarks, cfg.DATASET.CONFIDENCE_INVERT, num_folds=num_folds, pixel_to_mm_scale=pixel_to_mm_scale, combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS)


    #Get jaccard
    all_jaccard_data_dict = evaluate_jaccard(
        bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, num_folds=num_folds, combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS
    )
    all_jaccard_data = all_jaccard_data_dict["Jaccard All"]
   
    all_bins_concat_lms_sep_all_jacc = all_jaccard_data_dict["all jacc concat bins lms sep all"] # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list



    bound_return_dict= evaluate_bounds(
        bounds_all_lms, bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, num_folds, combine_middle_bins=cfg.PIPELINE.COMBINE_MIDDLE_BINS
    )

    all_bound_data = bound_return_dict["Error Bounds All"]
    all_bins_concat_lms_sep_all_errorbound = bound_return_dict["all errorbound concat bins lms sep all"] # same as all_bins_concat_lms_sep_foldwise but folds are flattened to a single list


    # exit()
    
    if interpret:
        save_location=None

        #If we have combined the middle bins, we are only displaying 3 bins (outer edges, and combined middle bins).
        if cfg.PIPELINE.COMBINE_MIDDLE_BINS: 
            num_bins_display = 3
        else:
            num_bins_display = num_bins




        #Set x_axis labels for following plots.
        x_axis_labels = [r"$B_{{{}}}$".format(num_bins_display + 1 - (i + 1)) for i in range(num_bins_display + 1)]


        #get error bounds

        if display_settings["errors"]:
            # mean error concat for each bin
            if cfg.OUTPUT.SAVE_FIGURES:
                if cfg.BOXPLOT.SAMPLES_AS_DOTS:
                    dotted_addition = "_dotted"
                else:
                    dotted_addition = "_undotted"
                save_location = os.path.join(save_folder, save_file_preamble+ dotted_addition + "_error_all_lms.pdf")

            box_plot_per_model(
                cmaps,
                all_bins_concat_lms_nosep_error,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Mean Error (mm)",
                num_bins=num_bins_display,
                turn_to_percent=False,
                show_sample_info=cfg.BOXPLOT.SHOW_SAMPLE_INFO_MODE,
                show_individual_dots = cfg.BOXPLOT.SAMPLES_AS_DOTS,
                y_lim=cfg.BOXPLOT.ERROR_LIM,
                to_log=True,
                save_path= save_location
            )

            if show_individual_landmark_plots:
                #plot the concatentated errors for each landmark seperately
                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_error):

                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:

                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(save_folder, save_file_preamble+ dotted_addition +"_error_lm_" + str(idx_l) + ".pdf")

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
                            show_individual_dots = cfg.BOXPLOT.SAMPLES_AS_DOTS,
                            y_lim=cfg.BOXPLOT.ERROR_LIM,
                            to_log=True,
                            save_path= save_location
                        )



            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble+ dotted_addition +"mean_error_folds_all_lms.pdf")
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
                save_path= save_location
            )

          
      
        # Plot Error Bound Accuracy

        if display_settings["error_bounds"]:
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble +"_errorbound_all_lms.pdf")

            box_plot(
                cmaps,
                all_bound_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Error Bound Accuracy (%)",
                num_bins=num_bins_display,
                save_path= save_location,
                y_lim=120,

            )

            if show_individual_landmark_plots:
                #plot the concatentated error bounds for each landmark seperately
                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_errorbound):
                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:

                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(save_folder, save_file_preamble +"_errorbound_lm_" + str(idx_l) + ".pdf")

                        box_plot(
                            cmaps,
                            lm_data,
                            uncertainty_error_pairs,
                            models_to_compare,
                            x_axis_labels=x_axis_labels,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Error Bound Accuracy (%)",
                            num_bins=num_bins_display,
                            save_path= save_location,
                            y_lim=120,
                        )

                        


        # Plot Jaccard Index
        if display_settings["jaccard"]:
            if cfg.OUTPUT.SAVE_FIGURES:
                save_location = os.path.join(save_folder, save_file_preamble +"_jaccard_all_lms.pdf")
            box_plot(
                cmaps,
                all_jaccard_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Jaccard Index (%)",
                num_bins=num_bins_display,
                y_lim=120,
                show_sample_info="None",
                save_path= save_location,
            )


            if show_individual_landmark_plots:

                #plot the jaccard index for each landmark seperately

                for idx_l, lm_data in enumerate(all_bins_concat_lms_sep_all_jacc):
                    if idx_l in ind_landmarks_to_show or ind_landmarks_to_show == [-1]:

                        if cfg.OUTPUT.SAVE_FIGURES:
                            save_location = os.path.join(save_folder, save_file_preamble +"jaccard_lm_" + str(idx_l) + ".pdf")

                        box_plot(
                            cmaps,
                            lm_data,
                            uncertainty_error_pairs,
                            models_to_compare,
                            x_axis_labels=x_axis_labels,
                            x_label="Uncertainty Thresholded Bin",
                            y_label="Jaccard Index (%)",
                            num_bins=num_bins_display,
                            y_lim=100,
                            save_path=save_location
                        )









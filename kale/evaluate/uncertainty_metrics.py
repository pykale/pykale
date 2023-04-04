import logging
import os

import numpy as np
import pandas as pd

from kale.prepdata.tabular_transform import apply_confidence_inversion


def evaluate_bounds(
    estimated_bounds,
    bin_predictions,
    uncertainty_pairs,
    num_bins,
    landmarks,
    num_folds=8,
    show_fig=False,
    combine_middle_bins=False,
):
    """
        Evaluate uncertainty estimation's ability to predict error bounds for its quantile bins.
        For each bin, we calculate the accuracy as the % of predictions in the bin which are between the estimated error bounds.
        We calculate the accuracy for each dictionary in the bin_predictions dict. For each bin, we calculate: a) the mean and
        std over all folds and all landmarks b) the mean and std for each landmark over all folds.

    Args:
        estimated_bounds (Dict): dict of Pandas Dataframes where each dataframe has est. error bounds for all uncertainty measures for a model,
        bin_predictions (Dict): dict of Pandas Dataframes where each dataframe has errors, predicted bins for all uncertainty measures for a model,
        uncertainty_pairs ([list]): list of lists describing the different uncert combinations to test,
        er_col (str): column name of error column in uncertainty results,
        num_bins (int): Number of quantile bins,
        landmarks (list) list of landmarks to measure uncertainty estimation,
        show_fig (bool): Show a figure depicting error bound accuracy (default=False).


    Returns:
        [Dict,Dict]: Pair of dicts with error bounds accuracies for all landmarks combined and landmarks seperated.
    """
    if combine_middle_bins:
        num_bins = 3

    # Initialise results dicts
    all_bound_percents = {}
    all_bound_percents_nolmsep = {}

    all_concat_errorbound_bins_lm_sep_foldwise = [{} for x in range(len(landmarks))]
    all_concat_errorbound_bins_lm_sep_all = [{} for x in range(len(landmarks))]

    # Loop over combinations of models (model) and uncertainty types (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        error_bounds = estimated_bounds[model + " Error Bounds"]

        for up in uncertainty_pairs:
            uncertainty_type = up[0]

            fold_learned_bounds_mean_lms = []
            fold_learned_bounds_mean_bins = [[] for x in range(num_bins)]
            fold_learned_bounds_bins_lmsnotsep = [[] for x in range(num_bins)]
            fold_all_bins_concat_lms_sep_foldwise = [[[] for y in range(num_bins)] for x in range(len(landmarks))]
            fold_all_bins_concat_lms_sep_all = [[[] for y in range(num_bins)] for x in range(len(landmarks))]
            for fold in range(num_folds):
                # Get the ids for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Uncertainty bins"]
                ]
                fold_bounds = strip_for_bound(
                    error_bounds[error_bounds["fold"] == fold][uncertainty_type + " Uncertainty bounds"].values
                )

                return_dict = bin_wise_bound_eval(
                    fold_bounds,
                    fold_errors,
                    fold_bins,
                    landmarks,
                    uncertainty_type,
                    num_bins=num_bins,
                    show_fig=show_fig,
                )
                fold_learned_bounds_mean_lms.append(return_dict["mean all lms"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_learned_bounds_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_learned_bounds_bins_lmsnotsep[idx_bin] = (
                        fold_learned_bounds_bins_lmsnotsep[idx_bin] + return_dict["mean all"][idx_bin]
                    )

                    for lm in range(len(landmarks)):
                        fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin] = (
                            fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin]
                            + return_dict["all bins concatenated lms seperated"][lm][idx_bin]
                        )
                        combined = (
                            fold_all_bins_concat_lms_sep_all[lm][idx_bin]
                            + return_dict["all bins concatenated lms seperated"][lm][idx_bin]
                        )

                        fold_all_bins_concat_lms_sep_all[lm][idx_bin] = combined

            # Reverses order so they are worst to best i.e. B5 -> B1
            all_bound_percents[model + " " + uncertainty_type] = fold_learned_bounds_mean_bins[::-1]
            all_bound_percents_nolmsep[model + " " + uncertainty_type] = fold_learned_bounds_bins_lmsnotsep[::-1]

            for lm in range(len(all_concat_errorbound_bins_lm_sep_foldwise)):
                all_concat_errorbound_bins_lm_sep_foldwise[lm][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_lms_sep_foldwise[lm]
                all_concat_errorbound_bins_lm_sep_all[lm][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_lms_sep_all[lm]

    return {
        "Error Bounds All": all_bound_percents,
        "all_bound_percents_nolmsep": all_bound_percents_nolmsep,
        "all errorbound concat bins lms sep foldwise": all_concat_errorbound_bins_lm_sep_foldwise,
        "all errorbound concat bins lms sep all": all_concat_errorbound_bins_lm_sep_all,
    }


def bin_wise_bound_eval(
    fold_bounds_all_lms, fold_errors, fold_bins, landmarks, uncertainty_type, num_bins=5, show_fig=False
):
    """
    Helper function for evaluate_bounds.

    Args:
        fold_bounds_all_lms ([list]):List of lists for estimated error bounds for each landmark.
        fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
        fold_bins (Pandas Dataframe): Pandas Dataframe of predicted quantile bins for this fold.
        landmarks (list) list of landmarks to measure uncertainty estimation,
        uncertainty_type (string): Name of uncertainty type to calculate accuracy for,
        num_bins (int): Number of quantile bins,
        show_fig (bool): Show a figure depicting error bound accuracy (default=False).


    Returns:
        [Dict]: Dict with error bound accuracy statistics.
    """
    all_lm_perc = []
    all_qs_perc = [[] for x in range(num_bins)]
    all_qs_size = [[] for x in range(num_bins)]

    all_qs_errorbound_concat_lms_sep = [[[] for y in range(num_bins)] for x in range(len(landmarks))]

    for i_lm, landmark in enumerate(landmarks):
        true_errors_lm = fold_errors[(fold_errors["landmark"] == landmark)][["uid", uncertainty_type + " Error"]]
        pred_bins_lm = fold_bins[(fold_errors["landmark"] == landmark)][["uid", uncertainty_type + " Uncertainty bins"]]

        # Zip to dictionary
        true_errors_lm = dict(zip(true_errors_lm.uid, true_errors_lm[uncertainty_type + " Error"]))
        pred_bins_lm = dict(zip(pred_bins_lm.uid, pred_bins_lm[uncertainty_type + " Uncertainty bins"]))

        # The error bounds are from B1 -> B5 i.e. best quantile of predictions to worst quantile of predictions
        fold_bounds = fold_bounds_all_lms[i_lm]

        # For each bin, see what % of landmarks are between the error bounds. If bin=0 then lower bound = 0, if bin=Q then no upper bound
        # Keep track of #samples in each bin for weighted mean.

        # turn dictionary of predicted bins into [[num_bins]] array
        pred_bins_keys = []
        pred_bins_errors = []
        for i in range(num_bins):
            inner_list_bin = list([key for key, val in pred_bins_lm.items() if str(i) == str(val)])
            inner_list_errors = []

            for id_ in inner_list_bin:
                inner_list_errors.append(list([val for key, val in true_errors_lm.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list_bin)

        bins_acc = []
        bins_sizes = []
        # key_groups = []
        for q in range((num_bins)):
            inner_bin_correct = 0

            inbin_errors = pred_bins_errors[q]

            for error in inbin_errors:
                if q == 0:
                    lower = 0
                    upper = fold_bounds[q]

                    if error <= upper and error > lower:
                        inner_bin_correct += 1

                elif q < (num_bins) - 1:
                    lower = fold_bounds[q - 1]
                    upper = fold_bounds[q]

                    if error <= upper and error > lower:
                        inner_bin_correct += 1

                else:
                    lower = fold_bounds[q - 1]
                    upper = 999999999999999999999999999999

                    if error > lower:
                        inner_bin_correct += 1

            if inner_bin_correct == 0:
                accuracy_bin = 0
            elif len(inbin_errors) == 0:
                accuracy_bin = 1
            else:
                accuracy_bin = inner_bin_correct / len(inbin_errors)
            bins_sizes.append(len(inbin_errors))
            bins_acc.append(accuracy_bin)

            all_qs_perc[q].append(accuracy_bin)
            all_qs_size[q].append(len(inbin_errors))
            all_qs_errorbound_concat_lms_sep[i_lm][q].append(accuracy_bin)

        # Weighted average over all bins
        weighted_mean_lm = 0
        total_weights = 0
        for l_idx in range(len(bins_sizes)):
            bin_acc = bins_acc[l_idx]
            bin_size = bins_sizes[l_idx]
            weighted_mean_lm += bin_acc * bin_size
            total_weights += bin_size
        weighted_av = weighted_mean_lm / total_weights
        all_lm_perc.append(weighted_av)

    # Weighted average for each of the quantile bins.
    weighted_av_binwise = []
    for binidx in range(len(all_qs_perc)):
        bin_accs = all_qs_perc[binidx]
        bin_asizes = all_qs_size[binidx]

        weighted_mean_bin = 0
        total_weights_bin = 0
        for l_idx in range(len(bin_accs)):
            b_acc = bin_accs[l_idx]
            b_siz = bin_asizes[l_idx]
            weighted_mean_bin += b_acc * b_siz
            total_weights_bin += b_siz

        # Avoid div by 0
        if weighted_mean_bin == 0 or total_weights_bin == 0:
            weighted_av_bin = 0
        else:
            weighted_av_bin = weighted_mean_bin / total_weights_bin
        weighted_av_binwise.append(weighted_av_bin)

    # No weighted average, just normal av
    normal_av_bin_wise = []
    for binidx in range(len(all_qs_perc)):
        bin_accs = all_qs_perc[binidx]
        normal_av_bin_wise.append(np.mean(bin_accs))

    return {
        "mean all lms": np.mean(all_lm_perc),
        "mean all bins": weighted_av_binwise,
        "mean all": all_qs_perc,
        "all bins concatenated lms seperated": all_qs_errorbound_concat_lms_sep,
    }


def strip_for_bound(string_):
    """
    Strip string into list of floats

    Args:
        string_ (string): A string of floats, seperated by commas.
    Returns:
        list: list of floats
    """

    bounds = []
    for entry in string_:
        entry = entry[1:-1]
        bounds.append([float(i) for i in entry.split(",")])
    return bounds


def evaluate_correlations(
    bin_predictions,
    uncertainty_error_pairs,
    cmaps,
    num_bins,
    conf_invert_info,
    num_folds=8,
    pixel_to_mm_scale=1,
    combine_middle_bins=False,
    save_path=None,
    to_log=False,
):
    from kale.interpret.uncertainty_quantiles import fit_line_with_ci

    """
        Calculates the correlation between error and uncertainty (default: spearmans rank) for each bin and for each landmark.

        Args:
            fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
            fold_uncertainty_values (Pandas Dataframe): Pandas Dataframe for uncertainty values for this fold.
            num_bins (int): Number of quantile bins,
            landmarks (list) list of landmarks to measure uncertainty estimation,
            uncertainty_key (string): Name of uncertainty type to calculate accuracy for,
            pixel_to_mm_scale (float): Scale factor to convert from pixels to mm.
            correlation_type (string): Type of correlation to use (default: "spearman").

        Returns:
            [Dict]: Dict with correlation statistics.
            Keep track of the following:
                1) Correlations for all bins together:
                    A) per fold per landmark
                    B) all folds joined per landmark
                    C) per fold for for all landmarks joined
                    D) all folds for all landmarks joined
                2) Correlations for each quantile seperately
                    A) per fold per landmark
                    B) all folds joined per landmark
                    C) per fold for for all landmarks joined
                    D) all folds for all landmarks joined

    """

    logger = logging.getLogger("qbin")
    # define dict to save correlations to.
    correlation_dict = {}

    num_bins_quantiles = num_bins
    # If we are combining the middle bins, we only have the 2 edge bins and the middle bins are combined into 1 bin.
    if combine_middle_bins:
        num_bins = 3

    # Loop over models (model) and uncertainty methods (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        correlation_dict[model] = {}
        for up in uncertainty_error_pairs:  # up = [pair name, error name , uncertainty name]
            uncertainty_type = up[0]
            logger.info("All folds correlation for Model: %s, Uncertainty type %s :" % (uncertainty_type, model))
            fold_errors = np.array(
                data_structs[(data_structs["Testing Fold"].isin(np.arange(num_folds)))][
                    [uncertainty_type + " Error"]
                ].values.tolist()
            ).flatten()

            fold_uncertainty_values = data_structs[(data_structs["Testing Fold"].isin(np.arange(num_folds)))][
                [uncertainty_type + " Uncertainty"]
            ]

            invert_uncert_bool = [x[1] for x in conf_invert_info if x[0] == uncertainty_type][0]
            if invert_uncert_bool:
                fold_uncertainty_values = apply_confidence_inversion(fold_uncertainty_values, up[2])

            fold_uncertainty_values = np.array(fold_uncertainty_values.values.tolist()).flatten()

            # Get the quantiles of the data w.r.t to the uncertainty values.
            # Get the true uncertianty quantiles
            sorted_uncertainties = np.sort(fold_uncertainty_values)

            quantiles = np.arange(1 / num_bins_quantiles, 1, 1 / num_bins_quantiles)[: num_bins_quantiles - 1]
            quantile_thresholds = [np.quantile(sorted_uncertainties, q) for q in quantiles]

            # If we are combining the middle bins, combine the middle lists into 1 list.
            if combine_middle_bins:
                quantile_thresholds = [quantile_thresholds[0], quantile_thresholds[-1]]

            # fit a piece-wise linear regression line between quantiles, and return correlation values.
            # os.makedirs(os.path.join(save_path, model, uncertainty_type), exist_ok=True)

            save_path_fig = (
                os.path.join(save_path, model + "_" + uncertainty_type + "_correlation_pwr_all_lms.pdf")
                if save_path
                else None
            )
            corr_dict = fit_line_with_ci(
                fold_errors,
                fold_uncertainty_values,
                quantile_thresholds,
                cmaps,
                pixel_to_mm=pixel_to_mm_scale,
                save_path=save_path_fig,
                to_log=to_log,
            )

            correlation_dict[model][uncertainty_type] = corr_dict

    return correlation_dict


def generate_summary_df(results_dictionary, cols_to_save, sheet_name, save_location):
    """
    Generates pandas dataframe with summary statistics.


    Args:
        ind_lms_results [[]]: A 2D list of all lm errors. A list for each landmark
        sdr_dicts ([Dict]): A list of dictionaries from the function
            localization_evaluation.success_detection_rate().
    Returns:pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                       orient='index')
        PandasDataframe: A dataframe with statsitics including mean error, std error of All and individual
            landmarks. Also includes the SDR detection rates.
            It should look like:
            df = {"Mean Er": {"All": 1, "L0": 1,...}, "std Er": {"All":1, "l0": 1, ...}, ...}

    """

    # Save data to csv files
    # with open(save_location, 'w') as csvfile:
    summary_dict = {}
    for [col_dict_key, col_save_name] in cols_to_save:
        for um in results_dictionary[col_dict_key].keys():
            # print("inner keys", rpiesults_dictionary[col_dict_key].keys())
            col_data = results_dictionary[col_dict_key][um]
            # Remove instances of None (which were added when the list was empty, rather than nan)
            summary_dict["All " + um + " " + col_save_name + " Mean"] = np.mean(
                [x for sublist in col_data for x in sublist if x is not None]
            )
            summary_dict["All " + um + " " + col_save_name + " Std"] = np.std(
                [x for sublist in col_data for x in sublist if x is not None]
            )
            for bin_idx, bin_data in enumerate(col_data):
                # print(um,col_dict_key, bin_idx, len(bin_data))
                summary_dict["B" + str(bin_idx + 1) + " " + um + " " + col_save_name + " Mean"] = np.mean(
                    [x for x in bin_data if x is not None]
                )
                summary_dict["B" + str(bin_idx + 1) + " " + um + " " + col_save_name + " Std"] = np.std(
                    [x for x in bin_data if x is not None]
                )

    pd_df = pd.DataFrame.from_dict(summary_dict, orient="index")

    with pd.ExcelWriter(save_location, engine="xlsxwriter") as writer:
        for n, df in (pd_df).items():
            # print(n, "AND", df)
            df.to_excel(writer, sheet_name=sheet_name)


def get_mean_errors(
    bin_predictions,
    uncertainty_pairs,
    num_bins,
    landmarks,
    num_folds=8,
    pixel_to_mm_scale=1,
    combine_middle_bins=False,
):
    """
        Evaluate uncertainty estimation's mean error of each bin
        For each bin, we calculate the mean localisation error for each landmark and for all landmarks.
        We calculate the mean error for each dictionary in the bin_predictions dict. For each bin, we calculate: a) the mean and
        std over all folds and all landmarks b) the mean and std for each landmark over all folds.

    Args:
        bin_predictions (Dict): dict of Pandas Dataframes where each dataframe has errors, predicted bins for all uncertainty measures for a model,
        uncertainty_pairs ([list]): list of lists describing the different uncert combinations to test,
        num_bins (int): Number of quantile bins,
        landmarks (list) list of landmarks to measure uncertainty estimation,
        num_folds (int): Number of folds,


    Returns:
        [Dict,Dict]: Pair of dicts with mean error for all landmarks combined and landmarks seperated.
        Keys that are returned:
            "all mean error bins nosep":  For every fold, the mean error for each bin. All landmarks are combined in the same list
            "all mean error bins lms sep":   For every fold, the mean error for each bin. Each landmark is in a seperate list
            "all error concat bins lms nosep":  For every fold, every error value in a list. Each landmark is in the same list. The list is flattened for all the folds.
            "all error concat bins lms sep foldwise":  For every fold, every error value in a list. Each landmark is in a seperate list. Each list has a list of results by fold.
            "all error concat bins lms sep all": For every fold, every error value in a list. Each landmark is in a seperate list. The list is flattened for all the folds.

    """
    # If we are combining the middle bins, we only have the 2 edge bins and the middle bins are combined into 1 bin.
    if combine_middle_bins:
        num_bins = 3

    # initialise empty dicts
    all_mean_error_bins = {}
    all_mean_error_bins_lms_sep = {}
    all_concat_error_bins_lm_sep_foldwise = [{} for x in range(len(landmarks))]
    all_concat_error_bins_lm_sep_all = [{} for x in range(len(landmarks))]

    all_concat_error_bins_lm_nosep = {}
    # Loop over models (model) and uncertainty methods (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        for up in uncertainty_pairs:  # up = [pair name, error name , uncertainty name]
            uncertainty_type = up[0]

            fold_mean_landmarks = []
            fold_mean_bins = [[] for x in range(num_bins)]
            fold_all_bins = [[] for x in range(num_bins)]
            fold_all_bins_concat_lms_sep_foldwise = [[[] for y in range(num_bins)] for x in range(len(landmarks))]
            fold_all_bins_concat_lms_sep_all = [[[] for y in range(num_bins)] for x in range(len(landmarks))]

            fold_all_bins_concat_lms_nosep = [[] for x in range(num_bins)]

            for fold in range(num_folds):
                # Get the errors and predicted bins for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Uncertainty bins"]
                ]
                # print("\n \n ", up, "fold: ", fold)
                return_dict = bin_wise_errors(
                    fold_errors, fold_bins, num_bins, landmarks, uncertainty_type, pixel_to_mm_scale=pixel_to_mm_scale
                )
                fold_mean_landmarks.append(return_dict["mean all lms"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_all_bins[idx_bin] = fold_all_bins[idx_bin] + return_dict["all bins"][idx_bin]

                    concat_no_sep = [x[idx_bin] for x in return_dict["all bins concatenated lms seperated"]]

                    flattened_concat_no_sep = [x for sublist in concat_no_sep for x in sublist]
                    flattened_concat_no_sep = [x for sublist in flattened_concat_no_sep for x in sublist]

                    fold_all_bins_concat_lms_nosep[idx_bin] = (
                        fold_all_bins_concat_lms_nosep[idx_bin] + flattened_concat_no_sep
                    )

                    for lm in range(len(landmarks)):
                        fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin] = (
                            fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin]
                            + return_dict["all bins concatenated lms seperated"][lm][idx_bin]
                        )

                        if return_dict["all bins concatenated lms seperated"][lm][idx_bin] != []:
                            combined = (
                                fold_all_bins_concat_lms_sep_all[lm][idx_bin]
                                + return_dict["all bins concatenated lms seperated"][lm][idx_bin][0]
                            )
                        else:
                            combined = fold_all_bins_concat_lms_sep_all[lm][idx_bin]

                        fold_all_bins_concat_lms_sep_all[lm][idx_bin] = combined

            # exit()
            # reverse orderings
            fold_mean_bins = fold_mean_bins[::-1]
            fold_all_bins = fold_all_bins[::-1]
            fold_all_bins_concat_lms_nosep = fold_all_bins_concat_lms_nosep[::-1]
            fold_all_bins_concat_lms_sep_foldwise = [x[::-1] for x in fold_all_bins_concat_lms_sep_foldwise]
            fold_all_bins_concat_lms_sep_all = [x[::-1] for x in fold_all_bins_concat_lms_sep_all]

            all_mean_error_bins[model + " " + uncertainty_type] = fold_mean_bins
            all_mean_error_bins_lms_sep[model + " " + uncertainty_type] = fold_all_bins

            all_concat_error_bins_lm_nosep[model + " " + uncertainty_type] = fold_all_bins_concat_lms_nosep

            for lm in range(len(fold_all_bins_concat_lms_sep_foldwise)):
                all_concat_error_bins_lm_sep_foldwise[lm][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_lms_sep_foldwise[lm]
                all_concat_error_bins_lm_sep_all[lm][model + " " + uncertainty_type] = fold_all_bins_concat_lms_sep_all[
                    lm
                ]

    return {
        "all mean error bins nosep": all_mean_error_bins,
        "all mean error bins lms sep": all_mean_error_bins_lms_sep,
        "all error concat bins lms nosep": all_concat_error_bins_lm_nosep,
        "all error concat bins lms sep foldwise": all_concat_error_bins_lm_sep_foldwise,
        "all error concat bins lms sep all": all_concat_error_bins_lm_sep_all,
    }


def evaluate_jaccard(bin_predictions, uncertainty_pairs, num_bins, landmarks, num_folds=8, combine_middle_bins=False):
    """
        Evaluate uncertainty estimation's ability to predict true error quantiles.
        For each bin, we calculate the jaccard index (JI) between the pred bins and GT error quantiles.
        We calculate the JI for each dictionary in the bin_predictions dict. For each bin, we calculate: a) the mean and
        std over all folds and all landmarks b) the mean and std for each landmark over all folds.

    Args:
        bin_predictions (Dict): dict of Pandas Dataframes where each dataframe has errors, predicted bins for all uncertainty measures for a model,
        uncertainty_pairs ([list]): list of lists describing the different uncert combinations to test,
        num_bins (int): Number of quantile bins,
        landmarks (list) list of landmarks to measure uncertainty estimation,
        num_folds (int): Number of folds,


    Returns:
        [Dict,Dict]: Pair of dicts with JI for all landmarks combined and landmarks seperated.
    """

    # If we are combining middle bins, we need the original number of bins to calcualate the true quantiles of the error.
    # Then, we combine all the middle bins of the true quantiles, giving us 3 bins.
    # There are only 3 bins that have been predicted, so set num_bins to 3.

    if combine_middle_bins:
        num_bins_for_quantiles = num_bins
        num_bins = 3
    else:
        num_bins_for_quantiles = num_bins
    # initialise empty dicts
    all_jaccard_data = {}
    all_jaccard_bins_lms_sep = {}

    all_recall_data = {}
    all_recall_bins_lms_sep = {}

    all_precision_data = {}
    all_precision__bins_lms_sep = {}

    all_concat_jacc_bins_lm_sep_foldwise = [{} for x in range(len(landmarks))]
    all_concat_jacc_bins_lm_sep_all = [{} for x in range(len(landmarks))]
    # Loop over models (model) and uncertainty methods (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        for up in uncertainty_pairs:  # up = [pair name, error name , uncertainty name]
            uncertainty_type = up[0]

            fold_mean_landmarks = []
            fold_mean_bins = [[] for x in range(num_bins)]
            fold_all_bins = [[] for x in range(num_bins)]

            fold_mean_landmarks_recall = []
            fold_mean_bins_recall = [[] for x in range(num_bins)]
            fold_all_bins_recall = [[] for x in range(num_bins)]

            fold_mean_landmarks_precision = []
            fold_mean_bins_precision = [[] for x in range(num_bins)]
            fold_all_bins_precision = [[] for x in range(num_bins)]

            fold_all_bins_concat_lms_sep_foldwise = [[[] for y in range(num_bins)] for x in range(len(landmarks))]
            fold_all_bins_concat_lms_sep_all = [[[] for y in range(num_bins)] for x in range(len(landmarks))]

            for fold in range(num_folds):
                # Get the errors and predicted bins for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Uncertainty bins"]
                ]

                return_dict = bin_wise_jaccard(
                    fold_errors,
                    fold_bins,
                    num_bins,
                    num_bins_for_quantiles,
                    landmarks,
                    uncertainty_type,
                    combine_middle_bins,
                )

                # print("Fodl: %s , BWJ: %s" % (fold, return_dict))
                fold_mean_landmarks.append(return_dict["mean all lms"])
                fold_mean_landmarks_recall.append(return_dict["mean all lms recall"])
                fold_mean_landmarks_precision.append(return_dict["mean all lms precision"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_all_bins[idx_bin] = fold_all_bins[idx_bin] + return_dict["all bins"][idx_bin]

                    fold_mean_bins_recall[idx_bin].append(return_dict["mean all bins recall"][idx_bin])
                    fold_all_bins_recall[idx_bin] = (
                        fold_all_bins_recall[idx_bin] + return_dict["all bins recall"][idx_bin]
                    )

                    fold_mean_bins_precision[idx_bin].append(return_dict["mean all bins precision"][idx_bin])
                    fold_all_bins_precision[idx_bin] = (
                        fold_all_bins_precision[idx_bin] + return_dict["all bins precision"][idx_bin]
                    )

                    # Get the jaccard saved for the individual landmarks, flattening the folds and also not flattening the folds
                    for lm in range(len(landmarks)):
                        fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin] = (
                            fold_all_bins_concat_lms_sep_foldwise[lm][idx_bin]
                            + return_dict["all bins concatenated lms seperated"][lm][idx_bin]
                        )
                        combined = (
                            fold_all_bins_concat_lms_sep_all[lm][idx_bin]
                            + return_dict["all bins concatenated lms seperated"][lm][idx_bin]
                        )

                        # if return_dict["all bins concatenated lms seperated"][lm][idx_bin] != []:
                        # combined = fold_all_bins_concat_lms_sep_all[lm][idx_bin] + return_dict["all bins concatenated lms seperated"][lm][idx_bin][0]
                        # else:
                        # combined = fold_all_bins_concat_lms_sep_all[lm][idx_bin]

                        fold_all_bins_concat_lms_sep_all[lm][idx_bin] = combined

            all_jaccard_data[model + " " + uncertainty_type] = fold_mean_bins
            all_jaccard_bins_lms_sep[model + " " + uncertainty_type] = fold_all_bins

            all_recall_data[model + " " + uncertainty_type] = fold_mean_bins_recall
            all_recall_bins_lms_sep[model + " " + uncertainty_type] = fold_all_bins_recall

            all_precision_data[model + " " + uncertainty_type] = fold_mean_bins_precision
            all_precision__bins_lms_sep[model + " " + uncertainty_type] = fold_all_bins_precision

            for lm in range(len(all_concat_jacc_bins_lm_sep_foldwise)):
                all_concat_jacc_bins_lm_sep_foldwise[lm][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_lms_sep_foldwise[lm]
                all_concat_jacc_bins_lm_sep_all[lm][model + " " + uncertainty_type] = fold_all_bins_concat_lms_sep_all[
                    lm
                ]

    return {
        "Jaccard All": all_jaccard_data,
        "Jaccard lms seperated": all_jaccard_bins_lms_sep,
        "Recall All": all_recall_data,
        "Recall lms seperated": all_recall_bins_lms_sep,
        "Precision All": all_precision_data,
        "Precision lms seperated": all_precision__bins_lms_sep,
        "all jacc concat bins lms sep foldwise": all_concat_jacc_bins_lm_sep_foldwise,
        "all jacc concat bins lms sep all": all_concat_jacc_bins_lm_sep_all,
    }


def bin_wise_errors(fold_errors, fold_bins, num_bins, landmarks, uncertainty_key, pixel_to_mm_scale):
    """
    Helper function for get_mean_errors. Calculates the mean error for each bin and for each landmark.

    Args:
        fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
        fold_bins (Pandas Dataframe): Pandas Dataframe of predicted quantile bins for this fold.
        num_bins (int): Number of quantile bins,
        landmarks (list) list of landmarks to measure uncertainty estimation,
        uncertainty_key (string): Name of uncertainty type to calculate accuracy for,


    Returns:
        [Dict]: Dict with mean error statistics.
    """

    all_lm_error = []
    all_qs_error = [[] for x in range(num_bins)]
    all_qs_error_concat_lms_sep = [[[] for y in range(num_bins)] for x in range(len(landmarks))]

    for i, landmark in enumerate(landmarks):
        true_errors_lm = fold_errors[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Error"]]
        pred_bins_lm = fold_bins[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Uncertainty bins"]]

        # Zip to dictionary
        true_errors_lm = dict(zip(true_errors_lm.uid, true_errors_lm[uncertainty_key + " Error"] * pixel_to_mm_scale))
        pred_bins_lm = dict(zip(pred_bins_lm.uid, pred_bins_lm[uncertainty_key + " Uncertainty bins"]))

        pred_bins_keys = []
        pred_bins_errors = []

        # This is saving them from best quantile of predictions to worst quantile of predictions in terms of uncertainty
        for j in range(num_bins):
            inner_list = list([key for key, val in pred_bins_lm.items() if str(j) == str(val)])
            inner_list_errors = []

            for id_ in inner_list:
                inner_list_errors.append(list([val for key, val in true_errors_lm.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list)

        # Now for each bin, get the mean error
        inner_errors = []
        for bin in range(num_bins):
            # pred_b_keys = pred_bins_keys[bin]
            pred_b_errors = pred_bins_errors[bin]

            # test for empty bin, it would've created a mean_error==nan , so don't add it!
            if pred_b_errors == []:
                continue

            mean_error = np.mean(pred_b_errors)

            # print(uncertainty_key, ": Bin %s and mean error %s +/- %s" % (bin, mean_error, np.std(pred_b_errors)))
            all_qs_error[bin].append(mean_error)
            all_qs_error_concat_lms_sep[i][bin].append(pred_b_errors)
            inner_errors.append(mean_error)

            # print(i, all_qs_error_concat_lms_sep[i])

        all_lm_error.append(np.mean(inner_errors))
    # print(all_qs_error_concat_lms_sep)
    # exit()

    # print("mean all lms ")
    # print("mal",mean_all_lms)
    # print("mean_all_bins ")
    # print("mab", mean_all_bins)
    mean_all_lms = np.mean(all_lm_error)
    mean_all_bins = []
    for x in all_qs_error:
        if x == []:
            mean_all_bins.append(None)
        else:
            mean_all_bins.append(np.mean(x))
    # mean_all_bins = [np.mean(x) for x in all_qs_error]

    return {
        "mean all lms": mean_all_lms,
        "mean all bins": mean_all_bins,
        "all bins": all_qs_error,
        "all bins concatenated lms seperated": all_qs_error_concat_lms_sep,
    }


def bin_wise_jaccard(
    fold_errors, fold_bins, num_bins, num_bins_quantiles, landmarks, uncertainty_key, combine_middle_bins
):
    """
    Helper function for evaluate_jaccard.

    Args:
        fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
        fold_bins (Pandas Dataframe): Pandas Dataframe of predicted quantile bins for this fold.
        num_bins (int): Number of quantile bins,
        landmarks (list) list of landmarks to measure uncertainty estimation,
        uncertainty_key (string): Name of uncertainty type to calculate accuracy for,


    Returns:
        [Dict]: Dict with JI statistics.
    """

    all_lm_jacc = []
    all_qs_jacc = [[] for x in range(num_bins)]

    all_qs_jacc_concat_lms_sep = [[[] for y in range(num_bins)] for x in range(len(landmarks))]

    all_lm_recall = []
    all_qs_recall = [[] for x in range(num_bins)]

    all_lm_precision = []
    all_qs_precision = [[] for x in range(num_bins)]

    for i, landmark in enumerate(landmarks):
        true_errors_lm = fold_errors[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Error"]]
        pred_bins_lm = fold_bins[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Uncertainty bins"]]

        # Zip to dictionary
        true_errors_lm = dict(zip(true_errors_lm.uid, true_errors_lm[uncertainty_key + " Error"]))
        pred_bins_lm = dict(zip(pred_bins_lm.uid, pred_bins_lm[uncertainty_key + " Uncertainty bins"]))

        pred_bins_keys = []
        pred_bins_errors = []

        # This is saving them from best quantile of predictions to worst quantile of predictions in terms of uncertainty
        for j in range(num_bins):
            inner_list = list([key for key, val in pred_bins_lm.items() if str(j) == str(val)])
            inner_list_errors = []

            for id_ in inner_list:
                inner_list_errors.append(list([val for key, val in true_errors_lm.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list)

        # Get the true error quantiles
        sorted_errors = [v for k, v in sorted(true_errors_lm.items(), key=lambda item: item[1])]

        quantiles = np.arange(1 / num_bins_quantiles, 1, 1 / num_bins_quantiles)[: num_bins_quantiles - 1]
        quantile_thresholds = [np.quantile(sorted_errors, q) for q in quantiles]

        # If we are combining the middle bins, combine the middle lists into 1 list.

        if combine_middle_bins:
            quantile_thresholds = [quantile_thresholds[0], quantile_thresholds[-1]]

        errors_groups = []
        key_groups = []

        for q in range(len(quantile_thresholds) + 1):
            inner_list_e = []
            inner_list_id = []
            for i_te, (id_, error) in enumerate(true_errors_lm.items()):
                if q == 0:
                    lower = 0
                    upper = quantile_thresholds[q]

                    if error <= upper:
                        inner_list_e.append(error)
                        inner_list_id.append(id_)

                elif q < len(quantile_thresholds):
                    lower = quantile_thresholds[q - 1]
                    upper = quantile_thresholds[q]

                    if error <= upper and error > lower:
                        inner_list_e.append(error)
                        inner_list_id.append(id_)

                else:
                    lower = quantile_thresholds[q - 1]
                    upper = 999999999999999999999999999999

                    if error > lower:
                        inner_list_e.append(error)
                        inner_list_id.append(id_)

            errors_groups.append(inner_list_e)
            key_groups.append(inner_list_id)

        # flip them so they go from B5 to B1
        pred_bins_keys = pred_bins_keys[::-1]
        pred_bins_errors = pred_bins_errors[::-1]
        errors_groups = errors_groups[::-1]
        key_groups = key_groups[::-1]
        # print("\n \n", uncertainty_key, ". qauntiles, ", quantiles, " and quantile thresholds: ", quantile_thresholds,)

        # print("sorted errors: ", sorted_errors)

        # Now for each bin, get the jaccard similarity
        inner_jaccard_sims = []
        inner_recalls = []
        inner_precisions = []
        for bin in range(num_bins):
            pred_b_keys = pred_bins_keys[bin]
            gt_bins_keys = key_groups[bin]

            j_sim = jaccard_similarity(pred_b_keys, gt_bins_keys)
            all_qs_jacc[bin].append(j_sim)
            all_qs_jacc_concat_lms_sep[i][bin].append(j_sim)

            inner_jaccard_sims.append(j_sim)

            # print("Bin %s, pred keys: %s, GT keys %s" % (bin, pred_b_keys, gt_bins_keys))
            # print("Pred len %s, GT len %s. Pred errors %s and GT errors %s" % (len(pred_bins_errors[bin]), len(errors_groups[bin]), np.sort(pred_bins_errors[bin]), np.sort(errors_groups[bin])))

            # If quantile threshold is the same as the last quantile threshold, the GT set is empty (rare, but can happen if distribution of errors is quite uniform).
            if len(gt_bins_keys) == 0:
                recall = 1
                precision = 0
            else:
                recall = sum(el in gt_bins_keys for el in pred_b_keys) / len(gt_bins_keys)

                if len(pred_b_keys) == 0 and len(gt_bins_keys) > 0:
                    precision = 0
                else:
                    precision = sum(1 for x in pred_b_keys if x in gt_bins_keys) / len(pred_b_keys)

            inner_recalls.append(recall)
            inner_precisions.append(precision)
            all_qs_recall[bin].append(recall)
            all_qs_precision[bin].append(precision)

        #     print("recall: ", recall, "precision: ", precision)

        # print("Inner jaccard sims: ", inner_jaccard_sims)
        all_lm_jacc.append(np.mean(inner_jaccard_sims))
        all_lm_recall.append(np.mean(inner_recalls))
        all_lm_precision.append(np.mean(inner_precisions))

    return {
        "mean all lms": np.mean(all_lm_jacc),
        "mean all bins": [np.mean(x) for x in all_qs_jacc],
        "all bins": all_qs_jacc,
        "mean all lms recall": np.mean(all_lm_recall),
        "mean all bins recall": [np.mean(x) for x in all_qs_recall],
        "all bins recall": all_qs_recall,
        "mean all lms precision": np.mean(all_lm_precision),
        "mean all bins precision": [np.mean(x) for x in all_qs_precision],
        "all bins precision": all_qs_precision,
        "all bins concatenated lms seperated": all_qs_jacc_concat_lms_sep,
    }


def jaccard_similarity(list1, list2):
    """
    Calculates the Jaccard Index (JI) between two lists.

    Args:
        list1 (list): list of set A,
        list2 (list): list of set B.
    Returns:
        float: JI between list1 and list2.
    """

    if len(list1) == 0 or len(list2) == 0:
        return 0
    else:
        intersection = len(set(list1).intersection(list2))  # no need to call list here
        union = len(list1 + list2) - intersection  # you only need to call len once here
        return intersection / union  # also no need to cast to float as this will be done for you

import numpy as np


def evaluate_bounds(
    estimated_bounds, bin_predictions, uncertainty_pairs, num_bins, landmarks, num_folds=8, show_fig=False,
):
    """Evaluate uncertainty estimation's ability to predict error bounds for its quantile bins.
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

    # Initialise results dicts
    all_bound_percents = {}
    all_bound_percents_nolmsep = {}

    # Loop over combinations of models (model) and uncertainty types (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        error_bounds = estimated_bounds[model + " Error Bounds"]

        for up in uncertainty_pairs:
            uncertainty_type = up[0]

            fold_learned_bounds_mean_lms = []
            fold_learned_bounds_mean_bins = [[] for x in range(num_bins)]
            fold_learned_bounds_bins_lmsnotsep = [[] for x in range(num_bins)]

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

            # Reverses order so they are worst to best i.e. B5 -> B1
            all_bound_percents[model + " " + uncertainty_type] = fold_learned_bounds_mean_bins[::-1]
            all_bound_percents_nolmsep[model + " " + uncertainty_type] = fold_learned_bounds_bins_lmsnotsep[::-1]
    return all_bound_percents, all_bound_percents_nolmsep


def bin_wise_bound_eval(
    fold_bounds_all_lms, fold_errors, fold_bins, landmarks, uncertainty_type, num_bins=5, show_fig=False
):
    """ Helper function for evaluate_bounds.
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

    for i, landmark in enumerate(landmarks):

        true_errors_lm = fold_errors[(fold_errors["landmark"] == landmark)][["uid", uncertainty_type + " Error"]]
        pred_bins_lm = fold_bins[(fold_errors["landmark"] == landmark)][["uid", uncertainty_type + " Uncertainty bins"]]

        # Zip to dictionary
        true_errors_lm = dict(zip(true_errors_lm.uid, true_errors_lm[uncertainty_type + " Error"]))
        pred_bins_lm = dict(zip(pred_bins_lm.uid, pred_bins_lm[uncertainty_type + " Uncertainty bins"]))

        # The error bounds are from B1 -> B5 i.e. best quantile of predictions to worst quantile of predictions
        fold_bounds = fold_bounds_all_lms[i]

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

        # Bounds chart for this fold, landmark
        if show_fig:
            raise ValueError("In Progress")
            # if len(np.unique(true_errors_lm)) == len(true_errors_lm) and weighted_av > 0.4:
            #     bounds_chart(true_errors_lm, pred_bins_errors)

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

    return {"mean all lms": np.mean(all_lm_perc), "mean all bins": weighted_av_binwise, "mean all": all_qs_perc}


def strip_for_bound(string_):
    """ Strip string into list of floats
    Args:
        string_ (string):string of floats, seperated by commas.
    Returns:
        list: list of floats
    """

    bounds = []
    for entry in string_:
        entry = entry[1:-1]
        bounds.append([float(i) for i in entry.split(",")])
    return bounds


def evaluate_jaccard(bin_predictions, uncertainty_pairs, num_bins, landmarks, num_folds=8):
    """Evaluate uncertainty estimation's ability to predict true error quantiles.
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

    # initialise empty dicts
    all_jaccard_data = {}
    all_jaccard_bins_lms_sep = {}

    # Loop over models (model) and uncertainty methods (up)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):

        for up in uncertainty_pairs:
            uncertainty_type = up[0]

            fold_mean_landmarks = []
            fold_mean_bins = [[] for x in range(num_bins)]
            fold_all_bins = [[] for x in range(num_bins)]

            for fold in range(num_folds):
                # Get the errors and predicted bins for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "landmark", uncertainty_type + " Uncertainty bins"]
                ]

                return_dict = bin_wise_jaccard(fold_errors, fold_bins, num_bins, landmarks, uncertainty_type)

                fold_mean_landmarks.append(return_dict["mean all lms"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_all_bins[idx_bin] = fold_all_bins[idx_bin] + return_dict["all bins"][idx_bin]

            all_jaccard_data[model + " " + uncertainty_type] = fold_mean_bins
            all_jaccard_bins_lms_sep[model + " " + uncertainty_type] = fold_all_bins

    return all_jaccard_data, all_jaccard_bins_lms_sep


def bin_wise_jaccard(fold_errors, fold_bins, num_bins, landmarks, uncertainty_key):
    """ Helper function for evaluate_jaccard.
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
    for i, landmark in enumerate(landmarks):

        true_errors_lm = fold_errors[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Error"]]
        pred_bins_lm = fold_bins[(fold_errors["landmark"] == landmark)][["uid", uncertainty_key + " Uncertainty bins"]]

        # Zip to dictionary
        true_errors_lm = dict(zip(true_errors_lm.uid, true_errors_lm[uncertainty_key + " Error"]))
        pred_bins_lm = dict(zip(pred_bins_lm.uid, pred_bins_lm[uncertainty_key + " Uncertainty bins"]))

        pred_bins_keys = []
        pred_bins_errors = []

        # This is saving them from worst quantile of predictions to best quantile of predictions in terms of uncertainty
        for i in range(num_bins):
            inner_list = list([key for key, val in pred_bins_lm.items() if str(i) == str(val)])
            inner_list_errors = []

            for id_ in inner_list:
                inner_list_errors.append(list([val for key, val in true_errors_lm.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list)

        # Get the true error quantiles
        sorted_errors = [v for k, v in sorted(true_errors_lm.items(), key=lambda item: item[1])]

        quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)
        quantile_thresholds = [np.quantile(sorted_errors, q) for q in quantiles]

        errors_groups = []
        key_groups = []
        for q in range(len(quantile_thresholds) + 1):
            inner_list_e = []
            inner_list_id = []
            for i, (id_, error) in enumerate(true_errors_lm.items()):
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

        # From worst quantile of errors to best quantile of errors
        # errors_groups = errors_groups[::-1]
        # key_groups = key_groups[::-1]

        # Now for each bin, get the jaccard similarity
        inner_jaccard_sims = []
        for bin in range(num_bins):
            pred_b_keys = pred_bins_keys[bin]
            gt_bins_keys = key_groups[bin]

            j_sim = jaccard_similarity(pred_b_keys, gt_bins_keys)
            all_qs_jacc[bin].append(j_sim)
            inner_jaccard_sims.append(j_sim)

        all_lm_jacc.append(np.mean(inner_jaccard_sims))

    return {
        "mean all lms": np.mean(all_lm_jacc),
        "mean all bins": [np.mean(x) for x in all_qs_jacc],
        "all bins": all_qs_jacc,
    }


def jaccard_similarity(list1, list2):
    """ Calculates the jaccard index between two lists.
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

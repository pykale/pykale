"""
Uncertainty Estimation for Landmark Localisaition


Reference:
Placeholder.html
"""

import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
from config import get_cfg_defaults

from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard
from kale.interpret.uncertainty_quantiles import box_plot, plot_cumulative, quantile_binning_and_est_errors
from kale.loaddata.tabular_access import load_uncertainty_pairs_csv
from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.prepdata.tabular_transform import apply_confidence_inversion, get_data_struct
from kale.utils.download import download_file_by_url


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Machine learning pipeline for PAH diagnosis")
    parser.add_argument("--cfg", required=False, help="path to config file", type=str)

    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    download_file_by_url(
        cfg.DATASET.SOURCE,
        cfg.DATASET.ROOT,
        "%s.%s" % (base_dir, cfg.DATASET.FILE_FORMAT),
        file_format=cfg.DATASET.FILE_FORMAT,
    )

    uncertainty_pairs_val = cfg.DATASET.UE_PAIRS_VAL
    uncertainty_pairs_test = cfg.DATASET.UE_PAIRS_TEST

    uncertainty_error_pairs = cfg.DATASET.UNCERTAINTY_ERROR_PAIRS
    models_to_compare = cfg.DATASET.MODELS
    dataset = cfg.DATASET.DATA
    landmarks = cfg.DATASET.LANDMARKS
    num_folds = cfg.DATASET.NUM_FOLDS

    num_bins = cfg.PIPELINE.NUM_QUANTILE_BINS

    save_folder = cfg.OUTPUT.SAVE_FOLDER

    # Define parameters for visualisation
    cmaps = sns.color_palette("deep", 10).as_hex()

    fit = True
    evaluate = True
    interpret = True

    # ---- This is the Fitting Phase ----
    if fit:
        for model in models_to_compare:
            for landmark in landmarks:

                # Define Paths for this loop
                landmark_results_path_val = os.path.join(
                    cfg.DATASET.ROOT, model, dataset, uncertainty_pairs_val + "_l" + str(landmark)
                )
                landmark_results_path_test = os.path.join(
                    cfg.DATASET.ROOT, model, dataset, uncertainty_pairs_test + "_l" + str(landmark)
                )

                uncert_boundaries, estimated_errors, predicted_bins = fit_and_predict(
                    model,
                    landmark,
                    uncertainty_error_pairs,
                    landmark_results_path_val,
                    landmark_results_path_test,
                    cfg,
                    save_folder,
                )

    ############ Evaluation Phase ##########################
    if evaluate:
        # saved_bins_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset, "res_predicted_bins")
        saved_bins_path_pre = os.path.join(save_folder, "Uncertainty_Preds")
        bins_all_lms, bins_lms_sep, bounds_all_lms, bounds_lms_sep = get_data_struct(
            models_to_compare, landmarks, saved_bins_path_pre, dataset
        )

        all_jaccard_data, all_jaccard_bins_lms_sep = evaluate_jaccard(
            bins_all_lms, uncertainty_error_pairs, num_bins, landmarks
        )
        all_bound_data, all_bound_lms_sep = evaluate_bounds(
            bounds_all_lms, bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, num_folds
        )

        if interpret:

            # Plot cumulative error figure for all predictions
            plot_cumulative(
                cmaps,
                bins_all_lms,
                models_to_compare,
                uncertainty_error_pairs,
                np.arange(num_bins),
                "Cumulative error for ALL predictions, dataset " + dataset,
                save_path=None,
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
                )

            x_axis_labels = [r"$B_{}$".format(num_bins + 1 - (i + 1)) for i in range(num_bins + 1)]

            # PLot Jaccard Index
            box_plot(
                cmaps,
                all_jaccard_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Jaccard Index (%)",
                num_bins=num_bins,
                y_lim=70,
            )

            # PLot Error Bound Accuracy
            box_plot(
                cmaps,
                all_bound_data,
                uncertainty_error_pairs,
                models_to_compare,
                x_axis_labels=x_axis_labels,
                x_label="Uncertainty Thresholded Bin",
                y_label="Error Bound Accuracy (%)",
                num_bins=num_bins,
            )


def fit_and_predict(model, landmark, uncertainty_error_pairs, ue_pairs_val, ue_pairs_test, config, save_folder=None):

    """ Loads (validation, testing data) pairs of (uncertainty, error) pairs and for each fold: used the validation
        set to generate quantile thresholds, estimate error bounds and bin the test data accordingly. Saves
        predicted bins and error bounds to a csv.

    Args:
        model (str): name of the model to perform uncertainty estimation on,
        landmark (int): Which landmark to perform uncertainty estimation on,
        uncertainty_error_pairs ([list]): list of lists describing the different uncert combinations to test,
        landmark_results_path (str): path to where the (error, uncertainty) pairs are saved,
        config (CfgNode): Config of hyperparameters.
        update_csv_w_fold (bool, optional): Whether to combine JSON and CSV files. Reccomended false (default=False),
        save_folder (str):path to folder to save results to *default=None).

    """

    num_bins = config.PIPELINE.NUM_QUANTILE_BINS
    invert_confidences = config.DATASET.CONFIDENCE_INVERT
    num_bins = config.PIPELINE.NUM_QUANTILE_BINS
    dataset = config.DATASET.DATA
    num_folds = config.DATASET.NUM_FOLDS

    # Save results across uncertainty pairings for each landmark.
    all_testing_results = pd.DataFrame(load_uncertainty_pairs_csv(ue_pairs_test, "Testing Fold", np.arange(num_folds)))
    error_bound_estimates = pd.DataFrame({"fold": np.arange(num_folds)})

    for idx, uncertainty_pairing in enumerate(uncertainty_error_pairs):

        uncertainty_category = uncertainty_pairing[0]
        invert_uncert_bool = [x[1] for x in invert_confidences if x[0] == uncertainty_category][0]
        uncertainty_localisation_er = uncertainty_pairing[1]
        uncertainty_measure = uncertainty_pairing[2]

        running_results = []
        running_error_bounds = []

        for fold in range(num_folds):

            validation_pairs = load_uncertainty_pairs_csv(
                ue_pairs_val, "Validation Fold", fold, ["uid", uncertainty_localisation_er, uncertainty_measure],
            )
            testing_pairs = load_uncertainty_pairs_csv(
                ue_pairs_test, "Testing Fold", fold, ["uid", uncertainty_localisation_er, uncertainty_measure]
            )

            if invert_uncert_bool:
                validation_pairs = apply_confidence_inversion(validation_pairs, uncertainty_measure)
                testing_pairs = apply_confidence_inversion(testing_pairs, uncertainty_measure)

            # Get Quantile Thresholds, fit IR line and estimate Error bounds. Return both and save for each fold and landmark.
            validation_ers = validation_pairs[uncertainty_localisation_er].values
            validation_uncerts = validation_pairs[uncertainty_measure].values
            uncert_boundaries, estimated_errors = quantile_binning_and_est_errors(
                validation_ers, validation_uncerts, num_bins, type="quantile"
            )

            # PREDICT for test data
            test_bins_pred = quantile_binning_predictions(
                dict(zip(testing_pairs.uid, testing_pairs[uncertainty_measure])), uncert_boundaries
            )
            running_results.append(test_bins_pred)
            running_error_bounds.append((estimated_errors))

        # Combine dictionaries and save if you want
        combined_dict_bins = {k: v for x in running_results for k, v in x.items()}

        all_testing_results[uncertainty_measure + " bins"] = list(combined_dict_bins.values())
        error_bound_estimates[uncertainty_measure + " bounds"] = running_error_bounds

    # Save Bin predictions and error bound estimations to spreadsheets
    if save_folder != None:
        save_bin_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset)
        os.makedirs(save_bin_path, exist_ok=True)
        all_testing_results.to_csv(
            os.path.join(save_bin_path, "res_predicted_bins_l" + str(landmark) + ".csv"), index=False
        )
        error_bound_estimates.to_csv(
            os.path.join(save_bin_path, "estimated_error_bounds_l" + str(landmark) + ".csv"), index=False
        )

    return uncert_boundaries, estimated_errors, all_testing_results


if __name__ == "__main__":
    main()

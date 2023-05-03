import logging
import os

import numpy as np
import pandas as pd

from kale.interpret.uncertainty_quantiles import quantile_binning_and_est_errors
from kale.loaddata.tabular_access import load_csv_columns
from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.prepdata.tabular_transform import apply_confidence_inversion


def fit_and_predict(
    landmark,
    uncertainty_error_pairs,
    ue_pairs_val,
    ue_pairs_test,
    num_bins,
    config,
    groundtruth_test_errors,
    save_folder=None,
):
    """Loads (validation, testing data) pairs of (uncertainty, error) pairs and for each fold: used the validation
        set to generate quantile thresholds, estimate error bounds and bin the test data accordingly. Saves
        predicted bins and error bounds to a csv.

    Args:
        landmark (int): Which landmark to perform uncertainty estimation on,
        uncertainty_error_pairs ([list]): list of lists describing the different uncert combinations to test,
        landmark_results_path (str): path to where the (error, uncertainty) pairs are saved,
        config (CfgNode): Config of hyperparameters.
        update_csv_w_fold (bool, optional): Whether to combine JSON and CSV files. Reccomended false (default=False),
        save_folder (str):path to folder to save results to *default=None).

    """

    logger = logging.getLogger("q_bin")
    invert_confidences = config.DATASET.CONFIDENCE_INVERT
    # dataset = config.DATASET.DATA
    num_folds = config.DATASET.NUM_FOLDS
    combine_middle_bins = config.PIPELINE.COMBINE_MIDDLE_BINS

    # Save results across uncertainty pairings for each landmark.
    all_testing_results = pd.DataFrame(load_csv_columns(ue_pairs_test, "Testing Fold", np.arange(num_folds)))
    error_bound_estimates = pd.DataFrame({"fold": np.arange(num_folds)})
    all_uncert_boundaries = pd.DataFrame({"fold": np.arange(num_folds)})

    for idx, uncertainty_pairing in enumerate(uncertainty_error_pairs):
        uncertainty_category = uncertainty_pairing[0]
        invert_uncert_bool = [x[1] for x in invert_confidences if x[0] == uncertainty_category][0]
        uncertainty_localisation_er = uncertainty_pairing[1]
        uncertainty_measure = uncertainty_pairing[2]

        running_results = []
        running_error_bounds = []
        running_uncert_boundaries = []

        for fold in range(num_folds):
            validation_pairs = load_csv_columns(
                ue_pairs_val, "Validation Fold", fold, ["uid", uncertainty_localisation_er, uncertainty_measure],
            )

            if groundtruth_test_errors:
                cols_to_get = ["uid", uncertainty_localisation_er, uncertainty_measure]
            else:
                cols_to_get = ["uid", uncertainty_measure]

            testing_pairs = load_csv_columns(ue_pairs_test, "Testing Fold", fold, cols_to_get)

            if invert_uncert_bool:
                validation_pairs = apply_confidence_inversion(validation_pairs, uncertainty_measure)
                testing_pairs = apply_confidence_inversion(testing_pairs, uncertainty_measure)

            # Get Quantile Thresholds, fit IR line and estimate Error bounds. Return both and save for each fold and landmark.
            validation_ers = validation_pairs[uncertainty_localisation_er].values
            validation_uncerts = validation_pairs[uncertainty_measure].values
            uncert_boundaries, estimated_errors = quantile_binning_and_est_errors(
                validation_ers, validation_uncerts, num_bins, type="quantile", combine_middle_bins=combine_middle_bins
            )

            # PREDICT for test data
            test_bins_pred = quantile_binning_predictions(
                dict(zip(testing_pairs.uid, testing_pairs[uncertainty_measure])), uncert_boundaries
            )
            running_results.append(test_bins_pred)
            running_error_bounds.append((estimated_errors))
            running_uncert_boundaries.append(uncert_boundaries)

        # Combine dictionaries and save if you want
        combined_dict_bins = {k: v for x in running_results for k, v in x.items()}

        all_testing_results[uncertainty_measure + " bins"] = list(combined_dict_bins.values())
        error_bound_estimates[uncertainty_measure + " bounds"] = running_error_bounds
        all_uncert_boundaries[uncertainty_measure + " bounds"] = running_uncert_boundaries
    # Save Bin predictions and error bound estimations to spreadsheets
    if save_folder is not None:
        save_bin_path = os.path.join(save_folder)
        os.makedirs(save_bin_path, exist_ok=True)
        all_testing_results.to_csv(
            os.path.join(save_bin_path, "res_predicted_bins_l" + str(landmark) + ".csv"), index=False
        )
        error_bound_estimates.to_csv(
            os.path.join(save_bin_path, "estimated_error_bounds_l" + str(landmark) + ".csv"), index=False
        )

        all_uncert_boundaries.to_csv(
            os.path.join(save_bin_path, "uncertainty_bounds_l" + str(landmark) + ".csv"), index=False
        )
        logger.info(
            "Saved predicted test bins for L%s, error bounds and uncertainty bounds to: %s", landmark, save_bin_path
        )
    return all_uncert_boundaries, error_bound_estimates, all_testing_results

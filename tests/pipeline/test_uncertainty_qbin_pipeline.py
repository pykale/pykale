import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns

import kale.utils.logger as logging
from examples.landmark_uncertainty.config import get_cfg_defaults
from kale.embed.uncertainty_fitting import fit_and_predict
from kale.interpret.uncertainty_quantiles import (
    generate_figures_comparing_bins,
    generate_figures_individual_bin_comparison,
)
from kale.utils.download import download_file_by_url

# DEFINE constants for testing
EXPECTED_FILES_IND_3 = [
    "3Bins/fitted_quantile_binningcumulative_error.pdf",
    "3Bins/fitted_quantile_binning/localisation_errors.xlsx",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_lm_0.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_errorbound_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_errorbound_lm_0.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_jaccard_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_recall_jaccard_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_precision_jaccard_all_lms.pdf",
    "3Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalsejaccard_lm_0.pdf",
]
# /home/schobs/Documents/results/testing/4CH/3Bins/
EXPECTED_FILES_IND_5 = [
    "5Bins/fitted_quantile_binningcumulative_error.pdf",
    "5Bins/fitted_quantile_binning/localisation_errors.xlsx",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_lm_0.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_errorbound_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_errorbound_lm_0.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_jaccard_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_recall_jaccard_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_precision_jaccard_all_lms.pdf",
    "5Bins/fitted_quantile_binning/testing_ind_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalsejaccard_lm_0.pdf",
]
EXPECED_FILES_FIT = [
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l2.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l2.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_l2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_l2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_l2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_l2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_l2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_l2.csv",
]
EXPECTED_FILES_COMP = [
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_error_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_error_lm_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_errorbound_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_errorbound_lm_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_recall_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_precision_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalsejaccard_lm_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_error_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_error_lm_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_errorbound_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_errorbound_lm_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_recall_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_precision_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalsejaccard_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_error_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_error_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_errorbound_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_errorbound_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_recall_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_precision_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalsejaccard_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_error_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_error_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undottedmean_error_folds_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_errorbound_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_errorbound_lm_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_recall_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_precision_jaccard_all_lms.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalsejaccard_lm_0.pdf",
]


def test_qbin_pipeline():
    """Test the uncertainty quantile binning pipeline."""

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./examples/landmark_uncertainty/configs/testing_config.yaml")
    # /pykale/examples/landmark_uncertainty/configs/testing_config.yaml
    cfg.freeze()
    #  /pykale/tests/pipeline/test_uncertainty_qbin_pipeline.py
    # ---- setup output ----
    os.makedirs(cfg.OUTPUT.SAVE_FOLDER, exist_ok=True)
    logger = logging.construct_logger("q_bin", cfg.OUTPUT.SAVE_FOLDER, log_to_terminal=True)
    logger.info(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR

    # download data if neccesary
    if cfg.DATASET.SOURCE is not None:
        logger.info("Downloading data...")
        data_file_name = "%s.%s" % (base_dir, cfg.DATASET.FILE_FORMAT)
        download_file_by_url(
            cfg.DATASET.SOURCE, cfg.DATASET.ROOT, data_file_name, file_format=cfg.DATASET.FILE_FORMAT,
        )
        logger.info("Data downloaded to %s!", cfg.DATASET.ROOT + base_dir)

    uncertainty_pairs_val = cfg.DATASET.UE_PAIRS_VAL
    uncertainty_pairs_test = cfg.DATASET.UE_PAIRS_TEST
    gt_test_error_available = cfg.DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE

    ind_q_uncertainty_error_pairs = cfg.PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS
    ind_q_models_to_compare = cfg.PIPELINE.INDIVIDUAL_Q_MODELS

    compare_q_uncertainty_error_pairs = cfg.PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS
    compare_q_models_to_compare = cfg.PIPELINE.COMPARE_Q_MODELS

    dataset = cfg.DATASET.DATA
    landmarks = cfg.DATASET.LANDMARKS
    num_folds = cfg.DATASET.NUM_FOLDS

    ind_landmarks_to_show = cfg.PIPELINE.IND_LANDMARKS_TO_SHOW

    pixel_to_mm_scale = cfg.PIPELINE.PIXEL_TO_MM_SCALE

    # Define parameters for visualisation
    cmaps = sns.color_palette("deep", 10).as_hex()

    show_individual_landmark_plots = cfg.PIPELINE.SHOW_IND_LANDMARKS

    for num_bins in cfg.PIPELINE.NUM_QUANTILE_BINS:
        # create the folder to save to
        save_folder = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, str(num_bins) + "Bins")

        # ---- This is the Fitting Phase ----
        # Fit all the options for the individual q selection and comparison q selection

        all_models_to_compare = np.unique(ind_q_models_to_compare + compare_q_models_to_compare)
        all_uncert_error_pairs_to_compare = np.unique(
            ind_q_uncertainty_error_pairs + compare_q_uncertainty_error_pairs, axis=0
        )

        for model in all_models_to_compare:
            for landmark in landmarks:
                # Define Paths for this loop
                landmark_results_path_val = os.path.join(
                    cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_val + "_l" + str(landmark)
                )
                landmark_results_path_test = os.path.join(
                    cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_test + "_l" + str(landmark)
                )

                fitted_save_at = os.path.join(save_folder, "fitted_quantile_binning", model, dataset)
                os.makedirs(save_folder, exist_ok=True)

                test_qbin_fit(
                    landmark,
                    all_uncert_error_pairs_to_compare,
                    landmark_results_path_val,
                    landmark_results_path_test,
                    num_bins,
                    cfg,
                    gt_test_error_available,
                    fitted_save_at,
                    model,
                )

        # Evaluation Phase ##########################

        comparisons_models = "_".join(ind_q_models_to_compare)

        comparisons_um = [str(x[0]) for x in ind_q_uncertainty_error_pairs]
        comparisons_um = "_".join(comparisons_um)

        save_file_preamble = "_".join(
            [
                cfg.OUTPUT.SAVE_PREPEND,
                "ind",
                dataset,
                comparisons_models,
                comparisons_um,
                "combined" + str(cfg.PIPELINE.COMBINE_MIDDLE_BINS),
            ]
        )

        generate_figures_individual_bin_comparison(
            data=[
                ind_q_uncertainty_error_pairs,
                ind_q_models_to_compare,
                dataset,
                landmarks,
                num_bins,
                cmaps,
                os.path.join(save_folder, "fitted_quantile_binning"),
                save_file_preamble,
                cfg,
                show_individual_landmark_plots,
                True,
                num_folds,
                ind_landmarks_to_show,
                pixel_to_mm_scale,
            ],
            display_settings={
                "cumulative_error": True,
                "errors": True,
                "jaccard": True,
                "error_bounds": True,
                "correlation": False,
            },
        )

        if num_bins == 3:
            exp = EXPECTED_FILES_IND_3
        else:
            exp = EXPECTED_FILES_IND_5

        search_dir = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, "/")

        test_expected_files_exist(exp, search_dir)

        # Now delete for memory
        for expected_file in exp:
            file_path = os.path.join(os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset), expected_file)
            os.remove(file_path)

    # If we are comparing bins against eachother, we need to wait until all the bins have been fitted.
    for c_model in compare_q_models_to_compare:
        for c_er_pair in compare_q_uncertainty_error_pairs:
            save_file_preamble = "_".join(
                [
                    cfg.OUTPUT.SAVE_PREPEND,
                    "compQ",
                    c_model,
                    c_er_pair[0],
                    dataset,
                    "combined" + str(cfg.PIPELINE.COMBINE_MIDDLE_BINS),
                ]
            )

            all_fitted_save_paths = [
                os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, str(x_bins) + "Bins", "fitted_quantile_binning")
                for x_bins in cfg.PIPELINE.NUM_QUANTILE_BINS
            ]

            hatch_type = "o" if "PHD-NET" == c_model else ""
            color = cmaps[0] if c_er_pair[0] == "S-MHA" else cmaps[1] if c_er_pair[0] == "E-MHA" else cmaps[2]
            save_folder_comparison = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, "ComparisonBins")
            os.makedirs(save_folder_comparison, exist_ok=True)

            logger.info("Comparison Q figures for: %s and %s ", c_model, c_er_pair)
            generate_figures_comparing_bins(
                data=[
                    c_er_pair,
                    c_model,
                    dataset,
                    landmarks,
                    cfg.PIPELINE.NUM_QUANTILE_BINS,
                    cmaps,
                    all_fitted_save_paths,
                    save_folder_comparison,
                    save_file_preamble,
                    cfg,
                    show_individual_landmark_plots,
                    True,
                    num_folds,
                    ind_landmarks_to_show,
                    pixel_to_mm_scale,
                ],
                display_settings={
                    "cumulative_error": True,
                    "errors": True,
                    "jaccard": True,
                    "error_bounds": True,
                    "hatch": hatch_type,
                    "colour": color,
                },
            )

    test_expected_files_exist(EXPECTED_FILES_COMP, os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset))

    # Now delete for memory
    for expected_file in EXPECTED_FILES_COMP:
        file_path = os.path.join(os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset), expected_file)
        os.remove(file_path)


def test_expected_files_exist(expected_files, save_folder):
    num_found = 0
    for expected_file in expected_files:
        for dirpath, _, _ in os.walk(save_folder):
            file_path = os.path.join(dirpath, expected_file)
            if os.path.isfile(file_path):
                num_found += 1
                break

    assert num_found == len(expected_files)


def test_paths_in_directory(paths, root_dir):
    """
    Test if all paths have the same root directory.

    Parameters:
        paths (list): A list of paths to check.
        root_dir (str): The root directory that all paths should be contained within.

    Raises:
        AssertionError: If any path is not contained within the root directory.

    Returns:
        None.
    """
    # Check if all paths are contained within the root directory
    assert all(
        os.path.abspath(path).startswith(os.path.abspath(root_dir)) for path in paths
    ), f"Not all paths are contained within {root_dir}"


def test_qbin_fit(
    landmark: int,
    all_uncert_error_pairs_to_compare: np.ndarray,
    landmark_results_path_val: str,
    landmark_results_path_test: str,
    num_bins: int,
    cfg: Any,
    gt_test_error_available: bool,
    save_folder: str,
    model: str,
) -> None:
    """
    Test the `fit_and_predict()` function with a specific landmark.

    Args:
        landmark (int): The landmark index to use for testing.
        all_uncert_error_pairs_to_compare (np.ndarray): An array of shape `(N, 2)` containing uncertainty-error
            pairs to use for testing.
        landmark_results_path_val (str): The path to the directory containing validation results for the
            specified landmark.
        landmark_results_path_test (str): The path to the directory containing test results for the specified
            landmark.
        num_bins (int): The number of bins to use for quantile binning.
        cfg (Any): A configuration object containing various settings for the algorithm.
        gt_test_error_available (bool): Whether or not ground truth test errors are available.
        save_folder (str): The path to the directory where intermediate results should be saved.

    Returns:
        None

    Raises:
        AssertionError: If any of the tests fail.

    This function tests the `fit_and_predict()` function with a specific landmark and set of inputs. It first creates
    a directory at the `save_folder` path if one does not already exist. It then calls `fit_and_predict()` with the
    specified inputs to generate predictions and associated uncertainty estimates. The function then tests the
    resulting `estimated_errors`, `uncert_boundaries`, and `predicted_bins` arrays using the `csv_equality_helper()`
    function.

    Note that this function assumes that there are ground truth test errors available for testing. If ground truth
    errors are not available, you should set `gt_test_error_available` to `False`.
    """
    # Create the `save_folder` directory if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Call `fit_and_predict()` with the specified inputs
    uncert_boundaries, estimated_errors, predicted_bins = fit_and_predict(
        landmark,
        all_uncert_error_pairs_to_compare,
        landmark_results_path_val,
        landmark_results_path_test,
        num_bins,
        cfg,
        groundtruth_test_errors=gt_test_error_available,
        save_folder=save_folder,
    )

    # Test `estimated_errors` using `csv_equality_helper()`
    csv_equality_helper(
        estimated_errors,
        "./tests/test_data/Uncertainty_tuples/" + model + "/4CH/" + str(num_bins) + "Bins_fit/estimated_error_bounds",
        landmark,
    )

    # Test `uncert_boundaries` using `csv_equality_helper()`
    csv_equality_helper(
        uncert_boundaries,
        "./tests/test_data/Uncertainty_tuples/" + model + "/4CH/" + str(num_bins) + "Bins_fit/uncertainty_bounds",
        landmark,
    )

    # Test `predicted_bins` using `csv_equality_helper()`
    csv_equality_helper(
        predicted_bins,
        "./tests/test_data/Uncertainty_tuples/" + model + "/4CH/" + str(num_bins) + "Bins_fit/res_predicted_bins",
        landmark,
    )


def csv_equality_helper(array, csv_preamble, landmark):
    """
    Test if a given numpy array is equal to a csv file with a specific landmark and path.

    Parameters:
        array (numpy.ndarray): The numpy array to test for equality with the CSV file.
        csv_preamble (str): The path preamble for the CSV file to be compared with the numpy array.
        landmark (int): The landmark index for the CSV file.

    Raises:
        AssertionError: If the numpy array and the CSV file do not match.

    Returns:
        None.
    """

    array = array.to_numpy()
    # Convert the DataFrame to a numpy array
    read_array = read_csv_landmark(csv_preamble, landmark)

    for idx, val in enumerate(read_array):
        for inner_idx, inner_val in enumerate(val):
            if inner_idx == 0:
                continue
            assert str(inner_val) == str(array[idx][inner_idx])


def read_csv_landmark(csv_preamble, landmark):
    """
    Read a CSV file with landmark-specific data.

    Args:
        csv_preamble (str): The path preamble to the directory containing the CSV file.
        landmark (int): The landmark index associated with the CSV file.

    Returns:
        np.ndarray: A numpy array containing the contents of the CSV file.

    Raises:
        FileNotFoundError: If the CSV file at the specified path cannot be found.
        ValueError: If the CSV file contains invalid data.

    This function reads a CSV file containing landmark-specific data. The path to the CSV file is constructed
    from the `csv_preamble` argument and the `landmark` argument. The CSV file should contain only numeric data,
    with one row per data point and one column per feature. The function reads the CSV file using the pandas
    `read_csv()` function and converts the resulting DataFrame to a numpy array using the `to_numpy()` method.
    If the CSV file cannot be found or contains invalid data, the function raises an appropriate error.
    """

    csv_path = os.path.join(csv_preamble + "_l" + str(landmark) + ".csv")
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert the DataFrame to a numpy array
    read_array = df.to_numpy()
    return read_array

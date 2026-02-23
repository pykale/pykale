import os
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
import pytest

import kale.utils.logger as logging
from kale.embed.uncertainty_fitting import fit_and_predict
from kale.interpret.uncertainty_quantiles import ComparingBinsConfig, QuantileBinningAnalyzer, QuantileBinningConfig


@pytest.fixture(scope="module")
def testing_cfg():
    config_params = {
        "DATASET": {
            "SOURCE": "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip",
            "ROOT": "tests/test_data",
            "BASE_DIR": "Uncertainty_tuples",
            "FILE_FORMAT": "zip",
            "CONFIDENCE_INVERT": [["S-MHA", True], ["E-MHA", True], ["E-CPV", False]],
            "DATA": "4CH",
            "LANDMARKS": [0, 1, 2],
            "NUM_FOLDS": 8,
            "GROUND_TRUTH_TEST_ERRORS_AVAILABLE": True,
            "UE_PAIRS_VAL": "uncertainty_pairs_valid",
            "UE_PAIRS_TEST": "uncertainty_pairs_test",
        },
        "PIPELINE": {
            "NUM_QUANTILE_BINS": [3, 5],
            "COMPARE_INDIVIDUAL_Q": True,
            "INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS": [
                ["S-MHA", "S-MHA Error", "S-MHA Uncertainty"],
                ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"],
            ],
            "INDIVIDUAL_Q_MODELS": ["U-NET", "PHD-NET"],
            "COMPARE_Q_VALUES": True,
            "COMPARE_Q_MODELS": ["PHD-NET", "U-NET"],
            "COMPARE_Q_UNCERTAINTY_ERROR_PAIRS": [
                ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"],
                ["S-MHA", "S-MHA Error", "S-MHA Uncertainty"],
            ],
            "COMBINE_MIDDLE_BINS": False,
            "PIXEL_TO_MM_SCALE": 1.0,
            "INDIVIDUAL_LANDMARKS_TO_SHOW": [0],
            "SHOW_INDIVIDUAL_LANDMARKS": True,
        },
        "IM_KWARGS": {"cmap": "gray"},
        "MARKER_KWARGS": {
            "marker": "o",
            "markerfacecolor": (1, 1, 1, 0.1),
            "markeredgewidth": 1.5,
            "markeredgecolor": "r",
        },
        "WEIGHT_KWARGS": {"markersize": 6, "alpha": 0.7},
        "BOXPLOT": {"SAMPLES_AS_DOTS": False, "ERROR_LIM": 256, "SHOW_SAMPLE_INFO_MODE": "Average"},
        "OUTPUT": {"SAVE_FOLDER": "tests/test_data", "SAVE_PREPEND": "testing", "SAVE_FIGURES": True},
    }

    yield config_params


# DEFINE constants for testing
EXPECTED_FILES_INDIVIDUAL_3 = [
    "3Bins/fitted_quantile_binning/all_predictions_cumulative_error.pdf",
    "3Bins/fitted_quantile_binning/b1_predictions_cumulative_error.pdf",
    "3Bins/fitted_quantile_binning/PHD-NET_b1_vs_all_cumulative_error.pdf",
    "3Bins/fitted_quantile_binning/U-NET_b1_vs_all_cumulative_error.pdf",
    "3Bins/fitted_quantile_binning/target_errors.xlsx",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_all_targets.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_target_0.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_errorbound_all_targets.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_errorbound_target_0.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_jaccard_all_targets.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "3Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_jaccard_target_0.pdf",
]
# /home/schobs/Documents/results/testing/4CH/3Bins/
EXPECTED_FILES_INDIVIDUAL_5 = [
    "5Bins/fitted_quantile_binning/all_predictions_cumulative_error.pdf",
    "5Bins/fitted_quantile_binning/b1_predictions_cumulative_error.pdf",
    "5Bins/fitted_quantile_binning/PHD-NET_b1_vs_all_cumulative_error.pdf",
    "5Bins/fitted_quantile_binning/U-NET_b1_vs_all_cumulative_error.pdf",
    "5Bins/fitted_quantile_binning/target_errors.xlsx",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_all_targets.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_error_target_0.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_errorbound_all_targets.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_errorbound_target_0.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_jaccard_all_targets.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "5Bins/fitted_quantile_binning/testing_individual_4CH_U-NET_PHD-NET_S-MHA_E-MHA_combinedFalse_undotted_jaccard_target_0.pdf",
]
EXPECED_FILES_FIT = [
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t0.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t1.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t2.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t2.csv",
    "3Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t0.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t1.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t2.csv",
    "3Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t0.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t1.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/res_predicted_bins_t2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/estimated_error_bounds_t2.csv",
    "5Bins/fitted_quantile_binning/PHD-NET/4CH/uncertainty_bounds_t2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t0.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t1.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/res_predicted_bins_t2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/estimated_error_bounds_t2.csv",
    "5Bins/fitted_quantile_binning/U-NET/4CH/uncertainty_bounds_t2.csv",
]
EXPECTED_FILES_COMP = [
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_errorbound_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_errorbound_target_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_E-MHA_4CH_combinedFalse_undotted_jaccard_target_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_errorbound_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_errorbound_target_0.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_PHD-NET_S-MHA_4CH_combinedFalse_undotted_jaccard_target_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_errorbound_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_errorbound_target_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_E-MHA_4CH_combinedFalse_undotted_jaccard_target_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_errorbound_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_errorbound_target_0.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_recall_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_precision_jaccard_all_targets.pdf",
    "ComparisonBins/testing_compQ_U-NET_S-MHA_4CH_combinedFalse_undotted_jaccard_target_0.pdf",
]


def _create_analyzer(config, display_settings):
    """Create and return a QuantileBinningAnalyzer instance.

    Args:
        config: Configuration object (QuantileBinningConfig or ComparingBinsConfig) containing
            analysis parameters like save folder, colormap, and display options.
        display_settings (dict): Dictionary specifying which plots to generate (cumulative_error,
            errors, jaccard, error_bounds, correlation, hatch).

    Returns:
        QuantileBinningAnalyzer: Configured analyzer instance ready for analysis.
    """
    analyzer_config = {
        "plot_samples_as_dots": config.plot_samples_as_dots,
        "show_sample_info": config.show_sample_info,
        "boxplot_error_lim": config.boxplot_error_lim,
        "boxplot_config": {"colormap": config.colormap, "hatch_type": display_settings.get("hatch", "")},
        "save_folder": config.save_folder,
        "save_file_preamble": config.save_file_preamble,
        "save_figures": config.save_figures,
        "interpret": config.interpret,
    }
    return QuantileBinningAnalyzer(
        config=analyzer_config,
        display_settings=display_settings,
    )


def _cleanup_test_files(expected_files, base_path):
    """Remove test files after validation.

    Args:
        expected_files (list): List of relative file paths to remove.
        base_path (str): Base directory path where the files are located.
    """
    for expected_file in expected_files:
        file_path = os.path.join(base_path, expected_file)
        os.remove(file_path)


def _run_fitting_phase(testing_cfg, num_bins, save_folder, all_models, all_uncert_error_pairs, landmarks):
    """Execute the fitting phase for all models and landmarks.

    Args:
        testing_cfg (dict): Test configuration dictionary containing dataset and pipeline settings.
        num_bins (int): Number of quantile bins to use for uncertainty quantification.
        save_folder (str): Directory path where fitted results will be saved.
        all_models (np.ndarray): Array of model names to process (e.g., ['U-NET', 'PHD-NET']).
        all_uncert_error_pairs (np.ndarray): Array of uncertainty-error pair configurations.
        landmarks (list): List of landmark indices to process.
    """
    for model in all_models:
        for landmark in landmarks:
            landmark_results_path_val = os.path.join(
                testing_cfg["DATASET"]["ROOT"],
                testing_cfg["DATASET"]["BASE_DIR"],
                model,
                testing_cfg["DATASET"]["DATA"],
                testing_cfg["DATASET"]["UE_PAIRS_VAL"] + "_t" + str(landmark),
            )
            landmark_results_path_test = os.path.join(
                testing_cfg["DATASET"]["ROOT"],
                testing_cfg["DATASET"]["BASE_DIR"],
                model,
                testing_cfg["DATASET"]["DATA"],
                testing_cfg["DATASET"]["UE_PAIRS_TEST"] + "_t" + str(landmark),
            )

            fitted_save_at = os.path.join(save_folder, "fitted_quantile_binning", model, testing_cfg["DATASET"]["DATA"])
            os.makedirs(save_folder, exist_ok=True)

            helper_test_qbin_fit(
                landmark,
                all_uncert_error_pairs,
                landmark_results_path_val,
                landmark_results_path_test,
                num_bins,
                testing_cfg,
                testing_cfg["DATASET"]["GROUND_TRUTH_TEST_ERRORS_AVAILABLE"],
                fitted_save_at,
                model,
            )


def _run_individual_bin_comparison(testing_cfg, num_bins, save_folder, colormap):
    """Execute individual bin comparison analysis.

    Args:
        testing_cfg (dict): Test configuration dictionary containing pipeline and output settings.
        num_bins (int): Number of quantile bins (3 or 5).
        save_folder (str): Directory path where analysis results will be saved.
        colormap (str): Name of the matplotlib colormap to use for visualization.

    Returns:
        list: List of expected output file paths based on the number of bins.
    """
    individual_q_uncertainty_error_pairs = testing_cfg["PIPELINE"]["INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS"]
    individual_q_models_to_compare = testing_cfg["PIPELINE"]["INDIVIDUAL_Q_MODELS"]

    save_file_preamble = "_".join(
        [
            testing_cfg["OUTPUT"]["SAVE_PREPEND"],
            "individual",
            testing_cfg["DATASET"]["DATA"],
            "_".join(individual_q_models_to_compare),
            "_".join([str(x[0]) for x in individual_q_uncertainty_error_pairs]),
            "combined" + str(testing_cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"]),
        ]
    )

    config = QuantileBinningConfig(
        uncertainty_error_pairs=individual_q_uncertainty_error_pairs,
        models=individual_q_models_to_compare,
        dataset=testing_cfg["DATASET"]["DATA"],
        target_indices=testing_cfg["DATASET"]["LANDMARKS"],
        num_bins=num_bins,
        combine_middle_bins=testing_cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"],
        confidence_invert=testing_cfg["DATASET"]["CONFIDENCE_INVERT"],
        show_individual_target_plots=testing_cfg["PIPELINE"]["SHOW_INDIVIDUAL_LANDMARKS"],
        individual_targets_to_show=testing_cfg["PIPELINE"]["INDIVIDUAL_LANDMARKS_TO_SHOW"],
        save_folder=os.path.join(save_folder, "fitted_quantile_binning"),
        save_file_preamble=save_file_preamble,
        save_figures=testing_cfg["OUTPUT"]["SAVE_FIGURES"],
        plot_samples_as_dots=testing_cfg["BOXPLOT"]["SAMPLES_AS_DOTS"],
        show_sample_info=testing_cfg["BOXPLOT"]["SHOW_SAMPLE_INFO_MODE"],
        boxplot_error_lim=testing_cfg["BOXPLOT"]["ERROR_LIM"],
        colormap=colormap,
        interpret=True,
        num_folds=testing_cfg["DATASET"]["NUM_FOLDS"],
        error_scaling_factor=testing_cfg["PIPELINE"]["PIXEL_TO_MM_SCALE"],
    )

    display_settings = {
        "cumulative_error": True,
        "errors": True,
        "jaccard": True,
        "error_bounds": True,
        "correlation": False,
        "hatch": "o",
    }

    analyzer = _create_analyzer(config, display_settings)
    analyzer.run_individual_bin_comparison(config)

    return EXPECTED_FILES_INDIVIDUAL_3 if num_bins == 3 else EXPECTED_FILES_INDIVIDUAL_5


def _run_comparing_bins_analysis(testing_cfg, c_model, c_er_pair, colormap):
    """Execute comparing bins analysis for a specific model and error pair.

    Args:
        testing_cfg (dict): Test configuration dictionary with dataset, pipeline, and output settings.
        c_model (str): Model name to analyze (e.g., 'U-NET' or 'PHD-NET').
        c_er_pair (list): Uncertainty-error pair configuration [name, error_key, uncertainty_key].
        colormap (str): Name of the matplotlib colormap to use for visualization.
    """
    save_file_preamble = "_".join(
        [
            testing_cfg["OUTPUT"]["SAVE_PREPEND"],
            "compQ",
            c_model,
            c_er_pair[0],
            testing_cfg["DATASET"]["DATA"],
            "combined" + str(testing_cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"]),
        ]
    )

    fitted_save_paths = [
        os.path.join(
            testing_cfg["OUTPUT"]["SAVE_FOLDER"],
            testing_cfg["DATASET"]["DATA"],
            str(x_bins) + "Bins",
            "fitted_quantile_binning",
        )
        for x_bins in testing_cfg["PIPELINE"]["NUM_QUANTILE_BINS"]
    ]

    save_folder_comparison = os.path.join(
        testing_cfg["OUTPUT"]["SAVE_FOLDER"], testing_cfg["DATASET"]["DATA"], "ComparisonBins"
    )
    os.makedirs(save_folder_comparison, exist_ok=True)

    comparing_config = ComparingBinsConfig(
        uncertainty_error_pair=c_er_pair,
        model=c_model,
        dataset=testing_cfg["DATASET"]["DATA"],
        targets=testing_cfg["DATASET"]["LANDMARKS"],
        q_values=testing_cfg["PIPELINE"]["NUM_QUANTILE_BINS"],
        fitted_save_paths=fitted_save_paths,
        combine_middle_bins=testing_cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"],
        show_individual_target_plots=testing_cfg["PIPELINE"]["SHOW_INDIVIDUAL_LANDMARKS"],
        individual_targets_to_show=testing_cfg["PIPELINE"]["INDIVIDUAL_LANDMARKS_TO_SHOW"],
        save_folder=save_folder_comparison,
        save_file_preamble=save_file_preamble,
        save_figures=testing_cfg["OUTPUT"]["SAVE_FIGURES"],
        plot_samples_as_dots=testing_cfg["BOXPLOT"]["SAMPLES_AS_DOTS"],
        show_sample_info=testing_cfg["BOXPLOT"]["SHOW_SAMPLE_INFO_MODE"],
        boxplot_error_lim=testing_cfg["BOXPLOT"]["ERROR_LIM"],
        colormap=colormap,
        interpret=True,
        num_folds=testing_cfg["DATASET"]["NUM_FOLDS"],
        error_scaling_factor=testing_cfg["PIPELINE"]["PIXEL_TO_MM_SCALE"],
    )

    hatch_type = "o" if "PHD-NET" == c_model else ""
    display_settings = {
        "cumulative_error": True,
        "errors": True,
        "jaccard": True,
        "error_bounds": True,
        "hatch": hatch_type,
    }

    analyzer = _create_analyzer(comparing_config, display_settings)
    analyzer.run_comparing_bins_analysis(comparing_config)


def test_qbin_pipeline(testing_cfg):
    """Test the uncertainty quantile binning pipeline."""
    # Setup
    os.makedirs(testing_cfg["OUTPUT"]["SAVE_FOLDER"], exist_ok=True)
    logger = logging.construct_logger("q_bin", testing_cfg["OUTPUT"]["SAVE_FOLDER"], log_to_terminal=True)

    dataset = testing_cfg["DATASET"]["DATA"]
    landmarks = testing_cfg["DATASET"]["LANDMARKS"]
    colormap = "Set1"

    # Get unique models and uncertainty-error pairs for fitting
    all_models = np.unique(testing_cfg["PIPELINE"]["INDIVIDUAL_Q_MODELS"] + testing_cfg["PIPELINE"]["COMPARE_Q_MODELS"])
    all_uncert_error_pairs = np.unique(
        testing_cfg["PIPELINE"]["INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS"]
        + testing_cfg["PIPELINE"]["COMPARE_Q_UNCERTAINTY_ERROR_PAIRS"],
        axis=0,
    )

    # Process each bin configuration
    for num_bins in testing_cfg["PIPELINE"]["NUM_QUANTILE_BINS"]:
        save_folder = os.path.join(testing_cfg["OUTPUT"]["SAVE_FOLDER"], dataset, str(num_bins) + "Bins")

        # Fitting phase
        _run_fitting_phase(testing_cfg, num_bins, save_folder, all_models, all_uncert_error_pairs, landmarks)

        # Individual bin comparison and validation
        expected_files = _run_individual_bin_comparison(testing_cfg, num_bins, save_folder, colormap)
        search_dir = os.path.join(testing_cfg["OUTPUT"]["SAVE_FOLDER"], dataset, "/")
        helper_test_expected_files_exist(expected_files, search_dir)

        # Cleanup
        _cleanup_test_files(expected_files, os.path.join(testing_cfg["OUTPUT"]["SAVE_FOLDER"], dataset))

    # Comparing bins analysis
    for c_model in testing_cfg["PIPELINE"]["COMPARE_Q_MODELS"]:
        for c_er_pair in testing_cfg["PIPELINE"]["COMPARE_Q_UNCERTAINTY_ERROR_PAIRS"]:
            logger.info("Comparison Q figures for: %s and %s ", c_model, c_er_pair)
            _run_comparing_bins_analysis(testing_cfg, c_model, c_er_pair, colormap)

    # Final validation and cleanup
    helper_test_expected_files_exist(EXPECTED_FILES_COMP, os.path.join(testing_cfg["OUTPUT"]["SAVE_FOLDER"], dataset))
    _cleanup_test_files(EXPECTED_FILES_COMP, os.path.join(testing_cfg["OUTPUT"]["SAVE_FOLDER"], dataset))


def helper_test_expected_files_exist(expected_files, save_folder):
    """
    Test whether all expected files are present in the specified directory or any of its subdirectories.

    Args:
        expected_files (list): A list of file names or paths that are expected to be found in the `save_folder` or its subdirectories.
        save_folder (str): The root directory in which to search for the expected files.

    Returns:
        None

    Raises:
        AssertionError: If any expected file is not found in the `save_folder` or its subdirectories.
    """
    not_found_files = []
    found_files = []

    for expected_file in expected_files:
        found = False
        for dirpath, _, _ in os.walk(save_folder):
            file_path = os.path.join(dirpath, expected_file)
            if os.path.isfile(file_path):
                found_files.append(expected_file)
                found = True
                break
        if not found:
            not_found_files.append(expected_file)

    assert len(not_found_files) == 0, (
        f"Expected files not found in the save folder ({len(not_found_files)}/{len(expected_files)} missing): "
        + str(not_found_files)
        + "\n Found files: "
        + str(found_files)
    )


def helper_test_qbin_fit(
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
    resulting `estimated_errors`, `uncertainty_boundaries`, and `predicted_bins` arrays using the `csv_equality_helper()`
    function.

    Note that this function assumes that there are ground truth test errors available for testing. If ground truth
    errors are not available, you should set `gt_test_error_available` to `False`.
    """
    # Create the `save_folder` directory if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Call `fit_and_predict()` with the specified inputs
    uncertainty_boundaries, estimated_errors, predicted_bins = fit_and_predict(
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
        os.path.join(cfg["DATASET"]["ROOT"], cfg["DATASET"]["BASE_DIR"], model)
        + "/4CH/"
        + str(num_bins)
        + "Bins_fit/estimated_error_bounds",
        landmark,
    )

    # Test `uncertainty_boundaries` using `csv_equality_helper()`
    csv_equality_helper(
        uncertainty_boundaries,
        os.path.join(cfg["DATASET"]["ROOT"], cfg["DATASET"]["BASE_DIR"], model)
        + "/4CH/"
        + str(num_bins)
        + "Bins_fit/uncertainty_bounds",
        landmark,
    )

    # Test `predicted_bins` using `csv_equality_helper()`
    csv_equality_helper(
        predicted_bins,
        os.path.join(cfg["DATASET"]["ROOT"], cfg["DATASET"]["BASE_DIR"], model)
        + "/4CH/"
        + str(num_bins)
        + "Bins_fit/res_predicted_bins",
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

    # Check if the lengths of the arrays are the same
    assert len(array) == len(read_array), f"Length mismatch: array={len(array)}, read_array={len(read_array)}"

    for i, (orig_list, read_list) in enumerate(zip(array, read_array)):
        # Check if the two arrays have the same length
        assert len(orig_list) == len(
            read_list
        ), f"Length mismatch: array[{i}]={len(orig_list)}, read_array[{i}]={len(read_list)}"

        for orig_value, read_value in zip(orig_list, read_list):
            # If the original value is a list, convert it to a flattened NumPy array.
            is_array = isinstance(orig_value, list)
            if is_array:
                orig_value = np.asarray(orig_value).ravel()

                # Convert the read value to a Numpy array in two steps
                # Step 1: Remove brackets and commas
                read_value = read_value.strip("[]").replace(",", "")

                # Step 2: Use np.fromstring to parse the numerical values, matching the type of original value
                for ch in "[],":
                    read_value = read_value.replace(ch, "")

                read_value = np.fromstring(read_value, sep=" ", like=orig_value)

            # For numerical (either a NumPy array or a Number) values, compare numerically.
            if is_array or isinstance(orig_value, Number):
                np.testing.assert_allclose(orig_value, read_value)
                continue

            # For non-numerical values, compare directly.
            assert orig_value == read_value


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

    csv_path = os.path.join(csv_preamble + "_t" + str(landmark) + ".csv")
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert the DataFrame to a numpy array
    read_array = df.to_numpy()
    return read_array

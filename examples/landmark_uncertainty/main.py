"""
Uncertainty Estimation for Landmark Localisaition


Reference:
Placeholder.html
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from config import get_cfg_defaults
from pandas import *

from kale.embed.uncertainty_fitting import fit_and_predict

warnings.filterwarnings("error")
import kale.utils.logger as logging
from kale.interpret.uncertainty_quantiles import (
    generate_figures_comparing_bins,
    generate_figures_individual_bin_comparison,
    quantile_binning_and_est_errors,
)
from kale.loaddata.tabular_access import load_csv_columns
from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.prepdata.tabular_transform import apply_confidence_inversion
from kale.utils.download import download_file_by_url


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Machine learning pipeline for PAH diagnosis")
    parser.add_argument("--cfg", required=False, help="path to config file", type=str)

    args = parser.parse_args()

    """Example:  python main.py --cfg /mnt/tale_shared/schobs/pykale/pykale/examples/landmark_uncertainty/configs/isbi_config.yaml"""
    return args


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ---- setup output ----
    os.makedirs(cfg.OUTPUT.SAVE_FOLDER, exist_ok=True)
    logger = logging.construct_logger("q_bin", cfg.OUTPUT.SAVE_FOLDER, log_to_terminal=True)
    logger.info(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR

    # download data if neccesary
    if cfg.DATASET.SOURCE != None:
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

    if gt_test_error_available:
        fit = True
        evaluate = True
        interpret = True
    else:
        fit = True
        evaluate = False
        interpret = False

    show_individual_landmark_plots = cfg.PIPELINE.SHOW_IND_LANDMARKS

    for num_bins in cfg.PIPELINE.NUM_QUANTILE_BINS:
        # create the folder to save to
        save_folder = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, str(num_bins) + "Bins")

        # ---- This is the Fitting Phase ----
        if fit:
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

                    uncert_boundaries, estimated_errors, predicted_bins = fit_and_predict(
                        landmark,
                        all_uncert_error_pairs_to_compare,
                        landmark_results_path_val,
                        landmark_results_path_test,
                        num_bins,
                        cfg,
                        groundtruth_test_errors=gt_test_error_available,
                        save_folder=fitted_save_at,
                    )

        ############ Evaluation Phase ##########################

        if evaluate:
            # Get results for each individual bin.
            if cfg.PIPELINE.COMPARE_INDIVIDUAL_Q:
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
                        interpret,
                        num_folds,
                        ind_landmarks_to_show,
                        pixel_to_mm_scale,
                    ],
                    display_settings={
                        "cumulative_error": True,
                        "errors": True,
                        "jaccard": True,
                        "error_bounds": True,
                        "correlation": True,
                    },
                )

            # If we are comparing bins against eachother, we need to wait until all the bins have been fitted.
            if cfg.PIPELINE.COMPARE_Q_VALUES and num_bins == cfg.PIPELINE.NUM_QUANTILE_BINS[-1]:
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
                            os.path.join(
                                cfg.OUTPUT.SAVE_FOLDER, dataset, str(x_bins) + "Bins", "fitted_quantile_binning"
                            )
                            for x_bins in cfg.PIPELINE.NUM_QUANTILE_BINS
                        ]

                        hatch_type = "o" if "PHD-NET" == c_model else ""
                        color = (
                            cmaps[0] if c_er_pair[0] == "S-MHA" else cmaps[1] if c_er_pair[0] == "E-MHA" else cmaps[2]
                        )
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
                                interpret,
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


if __name__ == "__main__":
    main()

"""
Uncertainty Estimation for Landmark Localization


Reference:
L. A. Schobs, A. J. Swift and H. Lu,
"Uncertainty Estimation for Heatmap-Based Landmark Localization,"
in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034,
April 2023, doi: 10.1109/TMI.2022.3222730.

Paper link: https://arxiv.org/abs/2203.02351
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from config import get_cfg_defaults
from pandas import *

import kale.utils.logger as logging
from kale.embed.uncertainty_fitting import fit_and_predict
from kale.interpret.uncertainty_quantiles import generate_fig_comparing_bins, generate_fig_individual_bin_comparison
from kale.utils.download import download_file_by_url

warnings.filterwarnings("error")


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Quantile Binning for landmark uncertainty estimation.")
    parser.add_argument("--cfg", required=False, help="path to config file", type=str)

    args = parser.parse_args()

    """Example:
    python main.py

    To use a custom config, or a config file provided in the configs folder:
    python main.py --cfg ../configs/isbi_config.yaml

    """
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

    # Define parameters for visualization
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
            # Fit all the options for the individual Q selection and comparison Q selection

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

                generate_fig_individual_bin_comparison(
                    data=[
                        ind_q_uncertainty_error_pairs,
                        ind_q_models_to_compare,
                        dataset,
                        landmarks,
                        num_bins,
                        cmaps,
                        os.path.join(save_folder, "fitted_quantile_binning"),
                        save_file_preamble,
                        cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"],
                        cfg["OUTPUT"]["SAVE_FIGURES"],
                        cfg["DATASET"]["CONFIDENCE_INVERT"],
                        cfg["BOXPLOT"]["SAMPLES_AS_DOTS"],
                        cfg["BOXPLOT"]["SHOW_SAMPLE_INFO_MODE"],
                        cfg["BOXPLOT"]["ERROR_LIM"],
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

            # If we are comparing bins against each other, we need to wait until all the bins have been fitted.
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
                        generate_fig_comparing_bins(
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
                                cfg["PIPELINE"]["COMBINE_MIDDLE_BINS"],
                                cfg["OUTPUT"]["SAVE_FIGURES"],
                                cfg["BOXPLOT"]["SAMPLES_AS_DOTS"],
                                cfg["BOXPLOT"]["SHOW_SAMPLE_INFO_MODE"],
                                cfg["BOXPLOT"]["ERROR_LIM"],
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
                                "color": color,
                            },
                        )


if __name__ == "__main__":
    main()

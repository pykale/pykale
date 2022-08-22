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
from pandas import *

from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard, get_mean_errors
from kale.interpret.uncertainty_quantiles import box_plot, box_plot_per_model, generate_figures_comparing_bins, generate_figures_individual_bin_comparison, plot_cumulative, quantile_binning_and_est_errors
from kale.loaddata.tabular_access import load_csv_columns
from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.prepdata.tabular_transform import apply_confidence_inversion, get_data_struct
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
    print(cfg)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    
    #download data if neccesary 
    if cfg.DATASET.SOURCE != None:
        download_file_by_url(
            cfg.DATASET.SOURCE,
            cfg.DATASET.ROOT,
            "%s.%s" % (base_dir, cfg.DATASET.FILE_FORMAT),
            file_format=cfg.DATASET.FILE_FORMAT,
        )

    uncertainty_pairs_val = cfg.DATASET.UE_PAIRS_VAL
    uncertainty_pairs_test = cfg.DATASET.UE_PAIRS_TEST

    ind_q_uncertainty_error_pairs = cfg.PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS
    ind_q_models_to_compare = cfg.PIPELINE.INDIVIDUAL_Q_MODELS

    compare_q_uncertainty_error_pairs = cfg.PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS
    compare_q_models_to_compare = cfg.PIPELINE.COMPARE_Q_MODELS

    dataset = cfg.DATASET.DATA
    landmarks = cfg.DATASET.LANDMARKS
    num_folds = cfg.DATASET.NUM_FOLDS

    # num_bins = cfg.PIPELINE.NUM_QUANTILE_BINS
    ind_landmarks_to_show = cfg.PIPELINE.IND_LANDMARKS_TO_SHOW

    pixel_to_mm_scale = cfg.PIPELINE.PIXEL_TO_MM_SCALE

  
   
    # Define parameters for visualisation
    cmaps = sns.color_palette("deep", 10).as_hex()

    fit = True
    evaluate = True
    interpret = True
    show_individual_landmark_plots = cfg.PIPELINE.SHOW_IND_LANDMARKS

   


    for num_bins in cfg.PIPELINE.NUM_QUANTILE_BINS:
        #create the folder to save to
        save_folder = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, str(num_bins)+"Bins")
       

    
        # ---- This is the Fitting Phase ----
        if fit:

            #Fit all the options for the individual q selection and comparison q selection
           

            all_models_to_compare = np.unique(ind_q_models_to_compare+ compare_q_models_to_compare)
            all_uncert_error_pairs_to_compare = np.unique(ind_q_uncertainty_error_pairs+ compare_q_uncertainty_error_pairs, axis=0)


            for model in all_models_to_compare:
                for landmark in landmarks:

                    # Define Paths for this loop
                    landmark_results_path_val = os.path.join(
                        cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_val + "_l" + str(landmark)
                    )
                    landmark_results_path_test = os.path.join(
                        cfg.DATASET.ROOT,  base_dir, model, dataset, uncertainty_pairs_test + "_l" + str(landmark)
                    )

                    fitted_save_at = os.path.join(save_folder, "fitted_quantile_binning", model)
                    os.makedirs(save_folder, exist_ok=True)

                    uncert_boundaries, estimated_errors, predicted_bins = fit_and_predict(
                        landmark,
                        all_uncert_error_pairs_to_compare,
                        landmark_results_path_val,
                        landmark_results_path_test,
                        num_bins,
                        cfg,
                        save_folder=fitted_save_at,
                    )
            
          
            

        ############ Evaluation Phase ##########################

        if evaluate:

            #Get results for each individual bin.
            if cfg.PIPELINE.COMPARE_INDIVIDUAL_Q:
                comparisons_models = "_".join(ind_q_models_to_compare)

                comparisons_um = [str(x[0]) for x in ind_q_uncertainty_error_pairs]
                comparisons_um = "_".join(comparisons_um)
       
                save_file_preamble = "_".join([cfg.OUTPUT.SAVE_PREPEND,"ind", dataset, comparisons_models,comparisons_um, "combined" + str(cfg.PIPELINE.COMBINE_MIDDLE_BINS)])
    
                generate_figures_individual_bin_comparison(
                    data=
                        [
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
                            pixel_to_mm_scale
                        ],
                    display_settings = {"cumulative_error": False, "errors": True, "jaccard": True, "error_bounds": True, "correlation": True},
                )
                

            #If we are comparing bins against eachother, we need to wait until all the bins have been fitted.
            if cfg.PIPELINE.COMPARE_Q_VALUES and num_bins == cfg.PIPELINE.NUM_QUANTILE_BINS[-1]:

                for c_model in compare_q_models_to_compare:
                    for c_er_pair in compare_q_uncertainty_error_pairs:
                        save_file_preamble = "_".join([cfg.OUTPUT.SAVE_PREPEND,"compQ",c_model ,c_er_pair[0], dataset,"combined" + str(cfg.PIPELINE.COMBINE_MIDDLE_BINS)])


                        all_fitted_save_paths = [os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, str(x_bins)+"Bins", "fitted_quantile_binning") for x_bins in cfg.PIPELINE.NUM_QUANTILE_BINS]
                        
                        hatch_type = "o" if "PHD-NET" == c_model else ""
                        color = cmaps[0] if c_er_pair[0]=="S-MHA" else cmaps[1] if c_er_pair[0]=="E-MHA" else cmaps[2] 
                        save_folder_comparison = os.path.join(cfg.OUTPUT.SAVE_FOLDER, dataset, "ComparisonBins")
                        os.makedirs(save_folder_comparison, exist_ok=True)

                        print("Comparison Q figures for: ", c_model, c_er_pair)
                        generate_figures_comparing_bins(
                            data=
                                [
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
                                    pixel_to_mm_scale
                                ],
                            display_settings = {"cumulative_error": True, "errors": True, "jaccard": True, "error_bounds": True, "hatch":hatch_type, "colour":color},
                        )

            

        
def fit_and_predict(landmark, uncertainty_error_pairs, ue_pairs_val, ue_pairs_test,num_bins,  config,  save_folder=None):

    """ Loads (validation, testing data) pairs of (uncertainty, error) pairs and for each fold: used the validation
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

    invert_confidences = config.DATASET.CONFIDENCE_INVERT
    dataset = config.DATASET.DATA
    num_folds = config.DATASET.NUM_FOLDS
    combine_middle_bins = config.PIPELINE.COMBINE_MIDDLE_BINS

    # Save results across uncertainty pairings for each landmark.
    all_testing_results = pd.DataFrame(load_csv_columns(ue_pairs_test, "Testing Fold", np.arange(num_folds)))
    error_bound_estimates = pd.DataFrame({"fold": np.arange(num_folds)})

    for idx, uncertainty_pairing in enumerate(uncertainty_error_pairs):

        uncertainty_category = uncertainty_pairing[0]
        invert_uncert_bool = [x[1] for x in invert_confidences if x[0] == uncertainty_category][0]
        uncertainty_localisation_er = uncertainty_pairing[1]
        uncertainty_measure = uncertainty_pairing[2]

        running_results = []
        running_error_bounds = []

        for fold in range(num_folds):

            validation_pairs = load_csv_columns(
                ue_pairs_val, "Validation Fold", fold, ["uid", uncertainty_localisation_er, uncertainty_measure],
            )
            testing_pairs = load_csv_columns(
                ue_pairs_test, "Testing Fold", fold, ["uid", uncertainty_localisation_er, uncertainty_measure]
            )


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

        # Combine dictionaries and save if you want
        combined_dict_bins = {k: v for x in running_results for k, v in x.items()}

        all_testing_results[uncertainty_measure + " bins"] = list(combined_dict_bins.values())
        error_bound_estimates[uncertainty_measure + " bounds"] = running_error_bounds

    # Save Bin predictions and error bound estimations to spreadsheets
    if save_folder != None:
        save_bin_path = os.path.join(save_folder)
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

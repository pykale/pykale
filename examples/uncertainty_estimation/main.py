"""
Uncertainty Estimation for Landmark Localisaition

 change below

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2021). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""


# There is a VERY important reason why these results don't match my paper. In the paper we use the validation set on the trained model
# to get the thresholds, in this example we are using the test results. therefore they are not calibrated to a particular model.

import argparse
import os
import csv
import seaborn as sns
import numpy as np
import pandas as pd



from config import get_cfg_defaults
from sklearn.model_selection import cross_validate

from kale.interpret import model_weights, visualize
from kale.loaddata.image_access import read_dicom_images
from kale.loaddata.tabular_access import load_uncertainty_pairs_csv, apply_confidence_inversion, update_csvs_with_folds, get_data_struct

from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url

from kale.interpret.uncertainty_quantiles import quantile_binning_and_est_errors, box_plot, plot_cumulative
from kale.predict.uncertainty_binning import quantile_binning_predictions
from kale.evaluate.uncertainty_metrics import evaluate_jaccard, evaluate_bounds

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
    # if args.cfg:
    #     cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    


    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR
    landmark_base_filename = cfg.DATASET.LANDMARK_FILE_BASE

    uncertainty_error_pairs = cfg.DATASET.UNCERTAINTY_ERROR_PAIRS
    models_to_compare = cfg.DATASET.MODELS
    dataset = cfg.DATASET.DATA
    landmarks = cfg.DATASET.LANDMARKS
    num_folds = cfg.DATASET.NUM_FOLDS
    fold_file_base = cfg.DATASET.FOLD_FILE_BASE
    invert_confidences = cfg.DATASET.CONFIDENCE_INVERT

    num_bins = cfg.PIPELINE.NUM_QUANTILE_BINS 

    update_csv_w_fold = False #Only True if results and split info are in diff files, only relevent for author

    save_folder = cfg.OUTPUT.SAVE_FOLDER 


    #Define parameters for visualisation 
    cmaps = sns.color_palette("deep", 10).as_hex()



  

    # Load, learn thresholds, predict, and save for each model/uncertainty/landmark
    
    fit = False
    evaluate = True
    interpret = True
    ################# This is the Fitting Phase ########################
    if fit:
        for model in models_to_compare:
            for landmark in landmarks:
                #Define Paths for this loop
                landmark_results_path = os.path.join(base_dir, "Results", model, dataset, landmark_base_filename+"_l"+str(landmark))
                uncert_boundaries, estimated_errors, predicted_bins = fit_and_predict(model, landmark, uncertainty_error_pairs, landmark_results_path, cfg, save_folder)


    ############ Evaluation Phase ##########################
    if evaluate:
        # saved_bins_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset, "res_predicted_bins")
        saved_bins_path_pre =  os.path.join(save_folder, "Uncertainty_Preds")
        bins_all_lms, bins_lms_sep,  bounds_all_lms, bounds_lms_sep = get_data_struct(models_to_compare, landmarks,  saved_bins_path_pre, dataset)

        all_jaccard_data, all_jaccard_bins_lms_sep = evaluate_jaccard(bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, verbose=True)
        all_bound_data, all_bound_lms_sep = evaluate_bounds(bounds_all_lms, bins_all_lms, uncertainty_error_pairs, num_bins, landmarks, num_folds, verbose=True)

        if interpret:
            plot_cumulative(cmaps, bins_all_lms, models_to_compare, uncertainty_error_pairs, np.arange(num_bins), save_path=None, verbose=False)

            x_axis_labels = [r'$B_{}$'.format(num_bins+1- (i+1)) for i in range(num_bins+1)]
            box_plot(cmaps, all_jaccard_data, uncertainty_error_pairs, models_to_compare,  x_axis_labels=x_axis_labels, 
                x_label="Uncertainty Thresholded Bin", y_label="Jaccard Index (%)", num_bins=num_bins, verbose=True)
            box_plot(cmaps, all_bound_data, uncertainty_error_pairs, models_to_compare,  x_axis_labels=x_axis_labels, 
                x_label="Uncertainty Thresholded Bin", y_label="Jaccard Index (%)", num_bins=num_bins, verbose=True)



def fit_and_predict(model, landmark, uncertainty_error_pairs, landmark_results_path, config, save_folder=None):

    num_bins = config.PIPELINE.NUM_QUANTILE_BINS 
    invert_confidences = config.DATASET.CONFIDENCE_INVERT
    num_bins = config.PIPELINE.NUM_QUANTILE_BINS 
    dataset = config.DATASET.DATA
    num_folds = config.DATASET.NUM_FOLDS

    #Save results across uncertainty pairings for each landmark.
    all_testing_results = pd.DataFrame(load_uncertainty_pairs_csv(landmark_results_path, "Testing Fold", np.arange(num_folds)))
    error_bound_estimates =  pd.DataFrame({'fold' : np.arange(num_folds)})

    for idx, uncertainty_pairing in enumerate(uncertainty_error_pairs):

        uncertainty_category = uncertainty_pairing[0]
        # print("uc", uncertainty_category)
        invert_uncert_bool = [x[1] for x in invert_confidences if x[0]==uncertainty_category]
        uncertainty_localisation_er = uncertainty_pairing[1]
        uncertainty_measure = uncertainty_pairing[2]
 

        # if update_csv_w_fold:
        #     all_jsons = [os.path.join(base_dir, "Data", dataset, 'fold_information', fold_file_base+"_fold"+str(fold)+'.json') for fold in range(num_folds)]
        #     update_csvs_with_folds(landmark_results_path, uncertainty_localisation_er, uncertainty_measure, all_jsons)

        running_results = []
        running_error_bounds = []

        for fold in range(num_folds):

            validation_pairs = load_uncertainty_pairs_csv(landmark_results_path, "Validation Fold", fold,  ["uid", uncertainty_localisation_er, uncertainty_measure])
            testing_pairs = load_uncertainty_pairs_csv(landmark_results_path, "Testing Fold", fold,  ["uid", uncertainty_localisation_er, uncertainty_measure])

            if invert_uncert_bool:
                validation_pairs =  apply_confidence_inversion(validation_pairs, uncertainty_measure)
                testing_pairs =  apply_confidence_inversion(testing_pairs, uncertainty_measure)
            
            #Get Quantile Thresholds, fit IR line and estimate Error bounds. Return both and save for each fold and landmark.
            validation_ers = validation_pairs[uncertainty_localisation_er].values
            validation_uncerts = validation_pairs[uncertainty_measure].values
            uncert_boundaries, estimated_errors = quantile_binning_and_est_errors(validation_ers, validation_uncerts, num_bins, type='quantile')


            #PREDICT for test data
            test_bins_pred = quantile_binning_predictions(dict(zip(testing_pairs.uid, testing_pairs[uncertainty_measure])), uncert_boundaries)
            running_results.append(test_bins_pred)
            running_error_bounds.append((estimated_errors))


        #Combine dictionaries and save if you want
        combined_dict_bins = {k:v for x in running_results for k,v in x.items()}

       
        all_testing_results[uncertainty_measure + " bins"] = list(combined_dict_bins.values()) 
        error_bound_estimates[uncertainty_measure  + " bounds"] =  running_error_bounds

    #Save Bin predictions and error bound estimations to spreadsheets
    if save_folder != None:
        save_bin_path = os.path.join(save_folder, "Uncertainty_Preds", model, dataset)
        os.makedirs(save_bin_path, exist_ok=True)
        all_testing_results.to_csv(os.path.join(save_bin_path,"res_predicted_bins_l"+str(landmark)+'.csv'), index=False)
        error_bound_estimates.to_csv(os.path.join(save_bin_path,"estimated_error_bounds_l"+str(landmark)+'.csv'), index=False)

    return uncert_boundaries, estimated_errors, all_testing_results




            # exit()

                #Predict on testing data
    #file structures: 
        #Data
            #dataset
                #Fold information
        #Results
            #dataset
                #Model
                    #Landmark csvs
        # So dataset/model/landmark0.csv, dataset/model2/landmark0.csv etc


    #Ask to specify:
        # MODEL_NAME
        # LANDMARKS e.g. [0,1,2] or [0,2]
        # DATASET
        # COMMON FILE PATH FOR LANDMARK CSV: which is MODEL/?????_landmark_x where ??? is their own input. csv file need to end with _landmark_x
                 #csv should be in format: u_id, then pairs of uncertianty, errors they want to compare
        # COLUMN PAIRINGS with name for each: {"S-MHA": ['S-MHA Error, 'S-MHA Uncertainty'], "E-MHA": ['E-MHA Error, 'E-MHA Uncertainty']}
        #Invert uncertainty measure boolean for each uncertainty pairing: {"S-MHA": True, "E-CPV": False}. False by default

        # NUM_FOLDS
        # Common file path for fold info files which is data/?????_fold_x where ??? is their own input. csv file need to end with _fold_x

        #NUM_BINS for how many quantile bins they want

        #SAVE_NAME for a file name to save predicted bins to


     #LOAD DATA
    #####Load csv for (error, uncertainty) pairs depending on config files. Also need to specify validation/test split for fitting... ####


    #********** PREDICT STAGE
    #Do a loop for each model
        # for each landmark, 
            #*Make a datastruct to save the predicted bins by using the LANDMARK_CSV to extract u_ids and COLUMNPAIRINGS to make predicted bin spaces
            #For each fold:
                #For each uncertainty pairing:
                    # IN MAIN:
                        #* select user arguments and load csv file
                        # (LOAD DATA) --> make a function that takes in column (pair_names, spreadsheet, fold_json, split_category_name, inverse_bool)
                                #  and returns 2D array of [errors, uncertainties] using the ids in the json from the category name (e.g. validation, test).
                                # if inverse_boolean is true apply x = (1/x+e) *!DONE!*
                        #use load data function to load VALIDATION and TEST pairings  *!DONE!*

                        #* get thresholds and predict error bounds for the validation set
                        # (PREDICT) --> port in generate_thresholds function to take in (validation_pairings, num_bins) and return quantile thresholds and error bounds.
                                    #aside: is this really predict? this is more learn? but doesn't fit into EMBED

                        #* Make predictions for each of the test set
                        # (PREDICT) --> port quantile_calibration_ecpv_bounded but take in (thresholds, error_bounds, test_pairs) and return predicted bin
                            # Update the outer loop in the MAIN with save the predicted bins, error bound bool, error and uncertainty and FOLD by their u_id
            #Save datastruct of predicted bins as a csv at model/save_name_landmarkX.csv
                   


    #********** EVALUATE STAGE
    #* LOAD DATA :
        #each function takes in (models to compare, landmarks to compare, uncertainty_types_to_compare) and returns:
            #--> returns the useful datastructs for each model as defined in my code.
            #--> error by bins data : For each model give back a list of errors per bin: Model All, Model L1, MOdel L2, model L3
            #--> all errors:  For each model give back all errors: Model All, Model L1, MOdel L2, model L3
            #--> all errors:  For each model give back all errors: Model All, Model L1, MOdel L2, model L3


    #Offer landmark specific or all landmarks together, offer all models together or individual models. 
    # The actual jaccard index/bound accuracy will work per sample, the figures will try to be generic as possible but its hard (first code in my version).

    #* Jaccard index for predicted bins and GT error quantiles
    # (EVALUATE) --> function that takes in (u_id_predicted bins, u_id_prediciton_errors num_folds) and returns the average JI and fold-wise bin-wise JI.
    # (INTERPRET) --> make the box plot for this


    #* Error bound acc for predicted error bounds
    # (EVALUATE) --> function that takes in (u_id_predicted_bins, u_id_error_bounds, num_folds) and returns the average EBA and fold-wise bin-wise EBA.
    # (INTERPRET) --> make the box plot for this

  


if __name__ == "__main__":
    main()

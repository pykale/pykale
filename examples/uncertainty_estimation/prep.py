"""
PAH Diagnosis from Cardiac MRI via a Multilinear PCA-based Pipeline

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2021). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from config import get_cfg_defaults
from sklearn.model_selection import cross_validate

from kale.interpret import model_weights, visualize
from kale.loaddata.image_access import read_dicom_images
from kale.loaddata.tabular_access import apply_confidence_inversion, load_uncertainty_pairs_csv
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
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

    # \fold_information

    # print("ii", invert_confidences)
    # Load, learn thresholds, predict, and save for each model/uncertainty/landmark
    for model in models_to_compare:
        for landmark in landmarks:
            for uncertainty_pairing in uncertainty_error_pairs:
                uncertainty_category = uncertainty_pairing[0]
                # print("uc", uncertainty_category)
                invert_uncert_bool = [x[1] for x in invert_confidences if x[0] == uncertainty_category]
                uncertainty_localisation_er = uncertainty_pairing[1]
                uncertainty_measure = uncertainty_pairing[2]

                # Define Paths for this loop
                landmark_results_path = os.path.join(
                    base_dir, "Results", model, dataset, landmark_base_filename + "_l" + str(landmark) + ".csv"
                )
                print("lm res path: ", landmark_results_path)
                for fold in range(num_folds):
                    fold_info_path = os.path.join(
                        base_dir, "Data", dataset, "fold_information", fold_file_base + "_fold" + str(fold) + ".json"
                    )

                    # Add the fold infos from the json into the CSV file
                    with open(fold_file_base, "r") as myfile:
                        json_data = myfile.read()

                    # parse file
                    json_dict = json.loads(json_data)
                    update_csv_w_fold(
                        landmark_results_path,
                        uncertainty_localisation_er,
                        uncertainty_measure,
                        fold_info_path,
                        json_dict,
                    )

                    # # rs\Lawrence Schobs\Documents\PhD\pykale\example_files\Data\4CH\fold_information
                    # validation_pairs, test_pairs = load_uncertainty_pairs_csv(landmark_results_path, uncertainty_localisation_er, uncertainty_measure, fold_info_path)

                    # if invert_uncert_bool:
                    #     validation_pairs, test_pairs =  apply_confidence_inversion(validation_pairs, test_pairs, uncertainty_measure)

    # file structures:
    # Data
    # dataset
    # Fold information
    # Results
    # dataset
    # Model
    # Landmark csvs
    # So dataset/model/landmark0.csv, dataset/model2/landmark0.csv etc

    # Ask to specify:
    # MODEL_NAME
    # LANDMARKS e.g. [0,1,2] or [0,2]
    # DATASET
    # COMMON FILE PATH FOR LANDMARK CSV: which is MODEL/?????_landmark_x where ??? is their own input. csv file need to end with _landmark_x
    # csv should be in format: u_id, then pairs of uncertianty, errors they want to compare
    # COLUMN PAIRINGS with name for each: {"S-MHA": ['S-MHA Error, 'S-MHA Uncertainty'], "E-MHA": ['E-MHA Error, 'E-MHA Uncertainty']}
    # Invert uncertainty measure boolean for each uncertainty pairing: {"S-MHA": True, "E-CPV": False}. False by default

    # NUM_FOLDS
    # Common file path for fold info files which is data/?????_fold_x where ??? is their own input. csv file need to end with _fold_x

    # NUM_BINS for how many quantile bins they want

    # SAVE_NAME for a file name to save predicted bins to

    # LOAD DATA
    #####Load csv for (error, uncertainty) pairs depending on config files. Also need to specify validation/test split for fitting... ####

    # ********** PREDICT STAGE
    # Do a loop for each model
    # for each landmark,
    # *Make a datastruct to save the predicted bins by using the LANDMARK_CSV to extract u_ids and COLUMNPAIRINGS to make predicted bin spaces
    # For each fold:
    # For each uncertainty pairing:
    # IN MAIN:
    # * select user arguments and load csv file
    # (LOAD DATA) --> make a function that takes in column (pair_names, spreadsheet, fold_json, split_category_name, inverse_bool)
    #  and returns 2D array of [errors, uncertainties] using the ids in the json from the category name (e.g. validation, test).
    # if inverse_boolean is true apply x = (1/x+e) *!DONE!*
    # use load data function to load VALIDATION and TEST pairings  *!DONE!*

    # * get thresholds and predict error bounds for the validation set
    # (PREDICT) --> port in generate_thresholds function to take in (validation_pairings, num_bins) and return quantile thresholds and error bounds.
    # aside: is this really predict? this is more learn? but doesn't fit into EMBED

    # * Make predictions for each of the test set
    # (PREDICT) --> port quantile_calibration_ecpv_bounded but take in (thresholds, error_bounds, test_pairs) and return predicted bin and bool of if it was in error bounds for each pair.
    # Update the outer loop in the MAIN with save the predicted bins, error bound bool, error and uncertainty and FOLD by their u_id
    # Save datastruct of predicted bins as a csv at model/save_name_landmarkX.csv

    # ********** EVALUATE STAGE
    # * LOAD DATA :
    # each function takes in (models to compare, landmarks to compare, uncertainty_types_to_compare) and returns:
    # --> returns the useful datastructs for each model as defined in my code.
    # --> error by bins data : For each model give back a list of errors per bin: Model All, Model L1, MOdel L2, model L3
    # --> all errors:  For each model give back all errors: Model All, Model L1, MOdel L2, model L3
    # --> all errors:  For each model give back all errors: Model All, Model L1, MOdel L2, model L3

    # Offer landmark specific or all landmarks together, offer all models together or individual models.
    # The actual jaccard index/bound accuracy will work per sample, the figures will try to be generic as possible but its hard (first code in my version).

    # * Jaccard index for predicted bins and GT error quantiles
    # (EVALUATE) --> function that takes in (u_id_predicted bins, u_id_prediciton_errors num_folds) and returns the average JI and fold-wise bin-wise JI.
    # (INTERPRET) --> make the box plot for this

    # * Error bound acc for predicted error bounds
    # (EVALUATE) --> function that takes in (u_id_predicted_bins, u_id_error_bounds, num_folds) and returns the average EBA and fold-wise bin-wise EBA.
    # (INTERPRET) --> make the box plot for this


if __name__ == "__main__":
    main()

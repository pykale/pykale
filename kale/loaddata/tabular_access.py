import logging
import os
from enum import Enum

import numpy as np
import pydicom
import torch
import pandas as pd
from torchvision import datasets, transforms

from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.mnistm import MNISTM
from kale.loaddata.multi_domain import MultiDomainAccess, MultiDomainImageFolder
from kale.loaddata.usps import USPS
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url

import json
import re


def load_uncertainty_pairs_csv(datapath,  split, fold, cols_to_return="All"):

    """Read csv file of data, return by split, fold and columns to return.

    Args:
        datapath (str): Path to csv file of uncertainty results,
        split (str): column name for split e.g. Validation, testing,
        split (str): column name for split e.g. Validation, testing,
        cols_to_return ([str]): Which columns to return (default="All").


    Returns:
        [pandas dataframe, pandas dataframe]: dataframe selected
    """ 

    #Load the uncertainty & error results
    datafame = pd.read_csv(datapath+'.csv', header=0)

    if cols_to_return== "All":
        cols_to_return = datafame.columns



    #Test if a single fold or list of folds
    if isinstance(fold,(list,pd.core.series.Series,np.ndarray)):
        return_data = (datafame.loc[datafame[split].isin(fold)]).loc[:, cols_to_return]
    else:
        return_data = (datafame.loc[datafame[split] == fold]).loc[:, cols_to_return]

    return return_data



def update_csvs_with_folds(datapath, er_col, uncert_col, all_fold_info_paths):

    """Read csv file of data, json file of val/test splits and save a CSV file with both data and splits.

    Args:
        datapath (str): Path to csv file of uncertainty results.
        er_col (str): column name of error column in uncertainty results
        uncert_col (str): column name of uncertainty measure column in uncertainty results
        fold_info_path (str): Path to fold information.


    Returns:
        [pandas dataframe, pandas dataframe]: dataframe of validation and testing uncertainty & error values for each uid
    """ 

    #Load the uncertainty & error results
    uncert_er_datafame = pd.read_csv(datapath+'.csv', header=0)
    uncert_er_datafame['Validation Fold'] = ""
    uncert_er_datafame['Testing Fold'] = ""

    for i, fold_info_path in enumerate(all_fold_info_paths):
        #Load the json file for this fold, it tells us the correct val and test ids for this fold.     
        with open(fold_info_path) as json_data:
            val_test_splits = json.load(json_data)
        
        #Get the ids for this fold

        validation_ids = [d['id'] for d in val_test_splits["validation"]]
        testing_ids = [d['id'] for d in val_test_splits["testing"]]

        #Update Rows for validation  and test fold info
        if "PHD-NET" in datapath:
            for v_id in validation_ids:
                uncert_er_datafame.loc[uncert_er_datafame['uid'].isin([v_id+'EHJ_220', v_id+'ASPIRE SSC_246', v_id+'EHJ_220', v_id+'ASPIRE SSC_120']), 'Validation Fold'] = i
            for te_id in testing_ids:
                uncert_er_datafame.loc[uncert_er_datafame['uid'].isin([te_id+'EHJ_220', te_id+'ASPIRE SSC_246', te_id+'EHJ_220', te_id+'ASPIRE SSC_120']), 'Testing Fold'] = i

        else:
            for v_id in validation_ids:
                uncert_er_datafame.loc[uncert_er_datafame['uid'] == v_id, 'Validation Fold'] = i
            for te_id in testing_ids:
                uncert_er_datafame.loc[uncert_er_datafame['uid'] == te_id, 'Testing Fold'] = i
       
    #Rename the u_ids to shorter version for phdnet
    if "PHD-NET" in datapath:
        #all ids
        all_ids = [d['id'] for d in val_test_splits["validation"] + val_test_splits["testing"] + val_test_splits["training"]]
        for a_id in all_ids:
            uncert_er_datafame.loc[uncert_er_datafame['uid'].isin([a_id+'EHJ_220', a_id+'ASPIRE SSC_246', a_id+'EHJ_220', a_id+'ASPIRE SSC_120']), 'uid'] = a_id

    #Save CSV with the append "_wFolds"
    which_lm = datapath.split('_')[-1]
    base_before =  "_".join(datapath.split('_')[:-1])

    uncert_er_datafame.to_csv(base_before+'_wFolds_' + which_lm +'.csv', index=False,) 

#Helper Regex for JSON splitting. Splits string in alphabetical and numerical chunks and returns the first 2 chunks
def split_uids_bynum(s):
    """ Helper Regex for JSON splitting. Splits string in alphabetical and numerical chunks and returns the first 2 chunks
    Args:
        s (string):string to split
    Returns:
        s: split string into numerical and alphabetical chunks.

    """
    return filter(None, re.split(r'(\d+)', s))[:2]

def apply_confidence_inversion(data, uncertainty_measure):
    """ Inverses a list of numbers, adds a small number to avoid 1/0.
    Args:
        data (Dict): dictionary of data to invert
        uncertainty_measure (string): key of dict to invert

    Returns:
        Dict: dict with inverted data.

    """
    data[uncertainty_measure] = (1/data[uncertainty_measure] + 0.0000000000001)
    
    return data




def get_data_struct(models_to_compare, landmarks, saved_bins_path_pre, dataset):
    """ Makes a dict of pandas dataframe used to evalauate uncertainty measures
        Args:
            models_to_compare (list): list of set models to add to datastruct,
            landmarks (list): list of landmarks to add to datastruct.
            saved_bins_path_pre (string): preamble to path of where the predicted quantile bins are saved.
            dataset (string): string of what dataset you're measuring.

        Returns:
            [Dict, Dict, Dict, Dict]: dictionaries of pandas dataframes for: a) all error & pred info, b) landmark
            seperated error & pred info c) all estimated error bound d) landmark seperated estimated error bounds.
    """ 

    data_structs = {}
    data_struct_sep = {}#

    data_struct_bounds = {}
    data_struct_bounds_sep = {}

    for model in models_to_compare:
        all_landmarks = []
        all_err_bounds = []

        for lm in landmarks:
            bin_pred_path =  os.path.join(saved_bins_path_pre, model, dataset, "res_predicted_bins_l" + str(lm))
            bin_preds = pd.read_csv(bin_pred_path+'.csv', header=0)
            bin_preds["landmark"] = lm

   
            error_bounds_path = os.path.join(saved_bins_path_pre, model, dataset, "estimated_error_bounds_l" + str(lm))
            error_bounds_pred = pd.read_csv(error_bounds_path+'.csv', header=0)
            error_bounds_pred["landmark"] = lm

            all_landmarks.append(bin_preds)
            all_err_bounds.append(error_bounds_pred)
            data_struct_sep[model + " L"+str(lm)] = bin_preds
            data_struct_bounds_sep[model + "Error Bounds L"+str(lm)] = error_bounds_pred

        data_structs[model] = pd.concat(all_landmarks, axis=0, ignore_index=True)
        data_struct_bounds[model + " Error Bounds"] = pd.concat(all_err_bounds, axis=0, ignore_index=True)

    return data_structs, data_struct_sep,  data_struct_bounds, data_struct_bounds_sep

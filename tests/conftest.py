# Global settings for tests. Run before any test
import csv
import logging
import os

import pytest
from scipy.io import loadmat

from kale.utils.download import download_file_by_url
from tests.data.landmark_uncertainty_data import (
    return_dummy_bin_predictions_l0,
    return_dummy_bin_predictions_l1,
    return_dummy_error_bounds_l0,
    return_dummy_error_bounds_l1,
    return_dummy_test_data,
    return_dummy_valid_data,
)

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    os.makedirs(path, exist_ok=True)
    return path


gait_url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"


@pytest.fixture(scope="session")
def gait(download_path):
    download_file_by_url(gait_url, download_path, "gait.mat", "mat")
    return loadmat(os.path.join(download_path, "gait.mat"))


@pytest.fixture(scope="session")
def office_path(download_path):
    path_ = os.path.join(download_path, "office")
    os.makedirs(path_, exist_ok=True)
    return path_


# Landmark Global fixtures
landmark_uncertainty_url = (
    "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
)


# Downloads and unzips remote file
@pytest.fixture(scope="session")
def landmark_uncertainty_tuples_path(download_path):
    path_ = os.path.join(download_path, "Uncertainty_tuples")
    os.makedirs(path_, exist_ok=True)

    download_file_by_url(landmark_uncertainty_url, path_, "Uncertainty_tuples.zip", "zip")
    valid_path = os.path.join(path_, "U-NET/SA/uncertainty_pairs_valid_l0")
    test_path = os.path.join(path_, "U-NET/SA/uncertainty_pairs_test_l0")

    return valid_path, test_path, path_


# Downloads and creates local dummy data
@pytest.fixture(scope="session")
def landmark_uncertainty_local_dummy(download_path):

    valid_path_ = os.path.join(download_path, "Uncertainty_tuples/dummy_valid_data")
    os.makedirs(valid_path_, exist_ok=True)

    test_path_ = os.path.join(download_path, "Uncertainty_tuples/dummy_test_data")
    os.makedirs(test_path_, exist_ok=True)

    dummy_validation_data = return_dummy_valid_data()
    dummy_test_data = return_dummy_test_data()

    # save dummy data
    zipped_valid = zip(*dummy_validation_data.values())
    zipped_test = zip(*dummy_test_data.values())

    with open(valid_path_ + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_validation_data.keys())
        writer.writerows(zipped_valid)

    # save dummy data
    with open(test_path_ + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_test_data.keys())
        writer.writerows(zipped_test)

    # return and save dummy test prediction data
    dum_path_pre = os.path.join(download_path, "Uncertainty_tuples/U-NET/SA/")
    os.makedirs(dum_path_pre, exist_ok=True)
    dum_er_path_l0 = os.path.join(dum_path_pre, "estimated_error_bounds_l0")
    dum_er_path_l1 = os.path.join(dum_path_pre, "estimated_error_bounds_l1")
    dum_pred_bin_path_l0 = os.path.join(dum_path_pre, "res_predicted_bins_l0")
    dum_pred_bin_path_l1 = os.path.join(dum_path_pre, "res_predicted_bins_l1")

    dummy_err_bounds_l0 = return_dummy_error_bounds_l0()
    dummy_err_bounds_l1 = return_dummy_error_bounds_l1()
    dummy_pred_bins_l0 = return_dummy_bin_predictions_l0()
    dummy_pred_bins_l1 = return_dummy_bin_predictions_l1()

    # save dummy data
    zip_dummy_err_bounds_l0 = zip(*dummy_err_bounds_l0.values())
    zip_dummy_err_bounds_l1 = zip(*dummy_err_bounds_l1.values())
    zip_dummy_pred_bins_l0 = zip(*dummy_pred_bins_l0.values())
    zip_dummy_pred_bins_l1 = zip(*dummy_pred_bins_l1.values())

    with open(dum_er_path_l0 + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_err_bounds_l0.keys())
        writer.writerows(zip_dummy_err_bounds_l0)

    with open(dum_er_path_l1 + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_err_bounds_l1.keys())
        writer.writerows(zip_dummy_err_bounds_l1)

    with open(dum_pred_bin_path_l0 + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_pred_bins_l0.keys())
        writer.writerows(zip_dummy_pred_bins_l0)

    with open(dum_pred_bin_path_l1 + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(dummy_pred_bins_l1.keys())
        writer.writerows(zip_dummy_pred_bins_l1)

    return valid_path_, test_path_, os.path.join(download_path, "Uncertainty_tuples")

# Global settings for tests. Run before any test
import os

import pytest
from scipy.io import loadmat

from kale.utils.download import download_file_by_url


@pytest.fixture(scope="session")
def download_path():
    path = os.path.dirname(os.path.dirname(os.getcwd()))
    path = os.path.join(path, "tests", "test_data")
    return path


gait_url = "https://github.com/pykale/data/raw/main/video_data/gait/gait_gallery_data.mat"


@pytest.fixture(scope="session")
def gait(download_path):
    download_file_by_url(gait_url, download_path, "gait.mat", "mat")
    return loadmat(os.path.join(download_path, "gait.mat"))

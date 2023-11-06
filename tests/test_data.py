import os

import pytest
from scipy.io import loadmat
from pathlib import Path
from yacs.config import CfgNode
from kale.loaddata.mnistm import MNISTM
from kale.loaddata.usps import USPS
from kale.prepdata.image_transform import get_transform

from kale.utils.download import download_file_by_url
from kale.loaddata.image_access import DigitDataset, DigitDatasetAccess, get_cifar, ImageAccess
from torchvision import datasets, transforms

# # @pytest.fixture(scope="session")
def download_path():
    path = os.path.join("tests", "test_data")
    os.makedirs(path, exist_ok=True)
    return path

download_path = download_path()


#tests/conftest.py
gait_url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"
landmark_uncertainty_url = (
    "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
)

download_file_by_url(gait_url, download_path, "gait.mat", "mat")
download_file_by_url(landmark_uncertainty_url, download_path, "Uncertainty_tuples.zip", "zip")



#tests/embed/test_factorization.py
baseline_url = "https://github.com/pykale/data/raw/main/videos/gait/mpca_baseline.mat"
download_file_by_url(baseline_url, download_path, "baseline.mat", "mat")








#tests/loaddata/test_video_access.py
#tests/pipeline/test_video_domain_adapter.py
url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

# dataset_root =  download_path + "/video_test_data/"
download_file_by_url(
        url=url,
        output_directory=download_path,
        output_file_name="video_test_data.zip",
        file_format="zip",
    )



#tests/pipeline/test_multiomics_trainer.py
binary_class_data_url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"
multi_class_data_url = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"

dataset_root = download_path + "/multiomics/trainer/binary_class/"
download_file_by_url(
        url=binary_class_data_url,
        output_directory=dataset_root,
        output_file_name="binary_class.zip",
        file_format="zip",
    )

dataset_root = download_path + "/multiomics/trainer/multi_class/"
download_file_by_url(
        url=multi_class_data_url,
        output_directory=dataset_root,
        output_file_name="multi_class.zip",
        file_format="zip",
    )


#tests/prepdata/test_image_transform.py
cmr_url = "https://github.com/pykale/data/raw/main/images/ShefPAH-179/SA_64x64_v2.0.zip"
download_file_by_url(cmr_url, download_path, "SA_64x64.zip", "zip")


#tests/prepdata/test_image_access.py
def office_path(download_path):
    path_ = os.path.join(download_path, "office")
    os.makedirs(path_, exist_ok=True)
    return path_

office_path = office_path(download_path)


datasets.MNIST(download_path, train=True, transform=get_transform("mnistm"), download=True)
MNISTM(download_path, train=True, transform=get_transform("mnistm"), download=True)
USPS(download_path, train=True, transform=get_transform("mnistm"), download=True)
datasets.SVHN(download_path, split="train", transform=get_transform("mnistm"), download=True)
datasets.SVHN(download_path, split="test", transform=get_transform("mnistm"), download=True)
datasets.CIFAR10(download_path, train=True, download=True, transform=get_transform("cifar", augment=True))
datasets.CIFAR100(download_path, train=True, download=True, transform=get_transform("cifar", augment=True))


OFFICE_DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
url = "https://github.com/pykale/data/raw/main/images/office"
for domain_ in OFFICE_DOMAINS:
    filename = "%s.zip" % domain_
    data_url = "%s/%s" % (url, filename)
    download_file_by_url(data_url, office_path, filename, "zip")


#tests/utils/test_download.py
PARAM = [
    "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_11_train.pkl;a.pkl;pkl",
    "https://github.com/pykale/data/raw/main/videos/video_test_data.zip;video_test_data.zip;zip",
]
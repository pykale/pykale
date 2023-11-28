from pathlib import Path

from tdc.multi_pred import DTI
from torchvision import datasets

from kale.loaddata.mnistm import MNISTM
from kale.loaddata.usps import USPS
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url


def download_path():
    path = Path("tests") / "test_data"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


path_test_data = download_path()


# Downloading gait gallery data for tests/conftest.py test
url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"
download_file_by_url(url, path_test_data, "gait.mat", "mat")
# Downloading Landmark Global fixtures data for tests/conftest.py test
url = "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
download_file_by_url(url, path_test_data, "Uncertainty_tuples.zip", "zip")


#
# Downloading MPCA data for tests/embed/test_factorization.py test
url = "https://github.com/pykale/data/raw/main/videos/gait/mpca_baseline.mat"
download_file_by_url(url, path_test_data, "baseline.mat", "mat")


# Downloading Test Video Data for tests/loaddata/test_video_access.py and tests/pipeline/test_video_domain_adapter.py test
url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
download_file_by_url(
    url=url, output_directory=path_test_data, output_file_name="video_test_data.zip", file_format="zip"
)


# Downloading Binary Class and Multi Class Multiomics Dataset for tests/pipeline/test_multiomics_trainer.py test
url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"
dataset_root = str(Path(path_test_data) / "multiomics/trainer/binary_class/")
download_file_by_url(
    url=url,
    output_directory=dataset_root,
    output_file_name="binary_class.zip",
    file_format="zip",
)
url = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"
dataset_root = str(Path(path_test_data) / "multiomics/trainer/multi_class/")
download_file_by_url(url=url, output_directory=dataset_root, output_file_name="multi_class.zip", file_format="zip")

# Downloading CMR Dataset for tests/prepdata/test_image_transform.py test
url = "https://github.com/pykale/data/raw/main/images/ShefPAH-179/SA_64x64_v2.0.zip"
download_file_by_url(url, path_test_data, "SA_64x64.zip", "zip")


# Downloading MNISTM, USPS, SVHN, CIFAR10, CIFAR100 for tests/loaddata/test_image_access.py test
datasets.MNIST(path_test_data, train=True, transform=get_transform("mnistm"), download=True)
MNISTM(path_test_data, train=True, transform=get_transform("mnistm"), download=True)
USPS(path_test_data, train=True, transform=get_transform("mnistm"), download=True)
datasets.SVHN(path_test_data, split="train", transform=get_transform("mnistm"), download=True)
datasets.SVHN(path_test_data, split="test", transform=get_transform("mnistm"), download=True)
datasets.CIFAR10(path_test_data, train=True, download=True, transform=get_transform("cifar", augment=True))
datasets.CIFAR100(path_test_data, train=True, download=True, transform=get_transform("cifar", augment=True))


# Downloading Office-31 Dataset for tests/loaddata/test_image_access.py test
office_path = str(Path(path_test_data) / "office")
OFFICE_DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
url = "https://github.com/pykale/data/raw/main/images/office"
for domain_ in OFFICE_DOMAINS:
    filename = "%s.zip" % domain_
    data_url = "%s/%s" % (url, filename)
    download_file_by_url(data_url, office_path, filename, "zip")


# Downloading Drug-Target Interaction (DTI) Datasets for tests/loaddata/test_tdc_datasets.py test
SOURCES = ["BindingDB_Kd", "BindingDB_Ki"]
for source_name in SOURCES:
    test_dataset = DTI(name=source_name, path=path_test_data)


# Downloading Omniglot Dataset for tests/loaddata/test_few_shot.py test
url = "https://github.com/pykale/data/raw/main/images/omniglot/omniglot_demo.zip"
download_file_by_url(url=url, output_directory=path_test_data, output_file_name="omniglot_demo.zip", file_format="zip")

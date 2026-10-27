import time
from pathlib import Path

from tdc.multi_pred import DTI
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN

from kale.loaddata.mnistm import MNISTM
from kale.loaddata.usps import USPS
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url


def download_path():
    path = Path("tests").joinpath("test_data")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def download_gait_gallery_data(save_path):
    # Downloading gait gallery data for tests/conftest.py test
    url = "https://github.com/pykale/data/raw/main/videos/gait/gait_gallery_data.mat"
    download_file_by_url(url, save_path, "gait.mat", "mat")


def download_landmark_global_data(save_path):
    # Downloading landmark global fixtures data for tests/conftest.py test
    url = "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
    download_file_by_url(url, save_path, "Uncertainty_tuples.zip", "zip")


def download_mpca_data(save_path):
    # Downloading MPCA data for tests/embed/test_factorization.py test
    url = "https://github.com/pykale/data/raw/main/videos/gait/mpca_baseline.mat"
    download_file_by_url(url, save_path, "baseline.mat", "mat")


def download_video_data(save_path):
    # Downloading video data for tests/loaddata/test_video_access.py and tests/pipeline/test_video_domain_adapter.py tests
    url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
    download_file_by_url(url, save_path, "video_test_data.zip", "zip")


def download_multiomics_data(save_path):
    # Downloading Binary Class and Multi Class Multiomics datasets for tests/pipeline/test_multiomics_trainer.py test
    url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"
    binary_save_path = Path(save_path).joinpath("multiomics/trainer/binary_class/")
    download_file_by_url(url, binary_save_path, "binary_class.zip", "zip")

    url = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"
    multi_save_path = Path(save_path).joinpath("multiomics/trainer/multi_class/")
    download_file_by_url(url, multi_save_path, "multi_class.zip", "zip")


def download_cmr_data(save_path):
    # Downloading CMR dataset for tests/prepdata/test_image_transform.py test
    url = "https://github.com/pykale/data/raw/main/images/ShefPAH-179/SA_64x64_v2.0.zip"
    download_file_by_url(url, save_path, "SA_64x64.zip", "zip")


def download_mnist_data(save_path):
    # Downloading MNIST datasets for tests/loaddata/test_image_access.py test
    MNIST(save_path, train=True, transform=get_transform("mnistm"), download=True)
    MNISTM(save_path, train=True, transform=get_transform("mnistm"), download=True)


def download_usps_data(save_path):
    # Downloading USPS dataset for tests/loaddata/test_image_access.py test
    USPS(save_path, train=True, transform=get_transform("mnistm"), download=True)


# Copies of the original Stanford files: a commit-pinned Hugging Face mirror, UC San Diego, then pinned archive captures.
# The first source avoids UCSD certificate-chain failures and archive rate limiting seen on Linux CI runners.
SVHN_HF_REVISION = "5a7debc5ed720cfc6bb0cfa01706a4de9bdad436"
SVHN_HF_BASE = f"https://huggingface.co/datasets/darcook/DCPV-Artifacts/resolve/{SVHN_HF_REVISION}/data/svhn"
SVHN_MIRRORS = {
    "train_32x32.mat": [
        f"{SVHN_HF_BASE}/train_32x32.mat?download=true",
        "https://cseweb.ucsd.edu/~weijian/static/datasets/svhn/train_32x32.mat",
        "https://web.archive.org/web/20241223210216id_/http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    ],
    "test_32x32.mat": [
        f"{SVHN_HF_BASE}/test_32x32.mat?download=true",
        "https://cseweb.ucsd.edu/~weijian/static/datasets/svhn/test_32x32.mat",
        "https://web.archive.org/web/20260203030601id_/http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
    ],
}
# The Internet Archive answers HTTP 429 for about a minute after a large download, longer than the downloader's own
# retries wait, so every mirror is tried again after a pause.
SVHN_MIRROR_ROUNDS = 3
SVHN_MIRROR_PAUSE_SECONDS = 60


def _try_svhn_mirrors(save_path, filename, md5, errors):
    """Try each mirror once; return True after the first verified download, recording each failure in ``errors``."""
    for url in SVHN_MIRRORS[filename]:
        try:
            download_file_by_url(url, save_path, filename, "mat", md5=md5)
            return True
        except Exception as error:
            errors.append(f"{url}: {error}")
    return False


def _download_svhn_file(save_path, filename, md5):
    """Download one SVHN file from the first mirror that serves it with torchvision's MD5."""
    errors = []
    for pause in [0] + [SVHN_MIRROR_PAUSE_SECONDS] * (SVHN_MIRROR_ROUNDS - 1):
        time.sleep(pause)
        if _try_svhn_mirrors(save_path, filename, md5, errors):
            return
    raise RuntimeError(
        f"Could not download {filename} with MD5 {md5} from any mirror in {SVHN_MIRROR_ROUNDS} rounds:\n"
        + "\n".join(errors)
    )


def download_svhn_data(save_path):
    # Downloading SVHN dataset for tests/loaddata/test_image_access.py test
    # Only CI provisioning avoids the Stanford host, which became unreachable; kale's SVHNDatasetAccess still downloads
    # through torchvision. A file that fails torchvision's MD5 is rejected on purpose rather than loaded, and the next
    # mirror is tried. torchvision then loads the verified file with download=False, so it never contacts Stanford.
    for split in ("train", "test"):
        _, filename, md5 = SVHN.split_list[split]
        _download_svhn_file(save_path, filename, md5)
        SVHN(save_path, split=split, transform=get_transform("mnistm"), download=False)


def download_cifar_data(save_path):
    # Downloading CIFAR10 and CIFAR100 datasets for tests/loaddata/test_image_access.py test
    CIFAR10(save_path, train=True, download=True, transform=get_transform("cifar", augment=True))
    CIFAR100(save_path, train=True, download=True, transform=get_transform("cifar", augment=True))


def download_office_data(save_path):
    # Downloading Office-31 dataset for tests/loaddata/test_image_access.py test
    office_save_path = Path(save_path).joinpath("office")
    url = "https://github.com/pykale/data/raw/main/images/office"
    for domain in ["amazon", "caltech", "dslr", "webcam"]:
        filename = "%s.zip" % domain
        data_url = "%s/%s" % (url, filename)
        download_file_by_url(data_url, office_save_path, filename, "zip")


def download_dti_data(save_path):
    # Downloading Drug-Target Interaction (DTI) datasets for tests/loaddata/test_tdc_datasets.py test
    for source_name in ["BindingDB_Kd", "BindingDB_Ki"]:
        _ = DTI(name=source_name, path=save_path)


def download_omniglot_data(save_path):
    # Downloading Omniglot Dataset for tests/loaddata/test_few_shot.py test
    url = "https://github.com/pykale/data/raw/main/images/omniglot/omniglot_demo.zip"
    download_file_by_url(url, save_path, "omniglot_demo.zip", "zip")


if __name__ == "__main__":
    path_to_test_data = download_path()

    download_gait_gallery_data(path_to_test_data)
    download_landmark_global_data(path_to_test_data)
    download_mpca_data(path_to_test_data)
    download_video_data(path_to_test_data)
    download_multiomics_data(path_to_test_data)
    download_cmr_data(path_to_test_data)
    download_mnist_data(path_to_test_data)
    download_usps_data(path_to_test_data)
    download_svhn_data(path_to_test_data)
    download_cifar_data(path_to_test_data)
    download_office_data(path_to_test_data)
    download_dti_data(path_to_test_data)
    download_omniglot_data(path_to_test_data)

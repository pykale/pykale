# ===============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Raivo Koot, rekoot1@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# ===============================================================================

"""Data downloading and compressed data extraction functions, Based on
https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
https://github.com/pytorch/pytorch/blob/master/torch/hub.py
"""

import logging
import os
from pathlib import Path

from torch.hub import download_url_to_file
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, extract_archive


def download_file_by_url(url, output_directory, output_file_name, file_format=None):
    """Download file/compressed file by url.

    Args:
        url (string): URL of the object to download
        output_directory (string, optional): Full path where object will be saved
                                             Abosolute path recommended. Relative path also works.
        output_file_name (string, optional): File name which object will be saved as
        file_format (string, optional): File format
                                For compressed file, support ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]

    Example: (Grab the raw link from GitHub. Notice that using "raw" in the URL.)
        >>> url = "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_04_train.pkl"
        >>> download_file_by_url(url, "data", "a.pkl", "pkl")

        >>> url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
        >>> download_file_by_url(url, "data", "video_test_data.zip", "zip")

    """

    output_directory = Path(output_directory).absolute()
    file = Path(output_directory).joinpath(output_file_name)
    if os.path.exists(file):
        logging.info("Skipping Download and Extraction")
        return
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Downloading and extracting {}.".format(output_file_name))
        download_and_extract_archive(url=url, download_root=output_directory, filename=output_file_name)
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Downloading {}.".format(output_file_name))
        download_url_to_file(url, file)
        logging.info("Datasets downloaded in {}".format(file))


def download_file_gdrive(id, output_directory, output_file_name, file_format=None):
    """Download file/compressed file by Google Drive id.

    Args:
        id (string): Google Drive file id of the object to download
        output_directory (string, optional): Full path where object will be saved
                                             Abosolute path recommended. Relative path also works.
        output_file_name (string, optional): File name which object will be saved as
        file_format (string, optional): File format
                                For compressed file, support ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]

    Example:
        >>> gdrive_id = "1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga"
        >>> download_file_gdrive(gdrive_id, "data", "demo_datasets.zip", "zip")

        >>> gdrive_id = "1SV7fmAnWj-6AU9X5BGOrvGMoh2Gu9Nih"
        >>> download_file_gdrive(gdrive_id, "data", "dummy_data.csv", "csv")
    """

    output_directory = Path(output_directory).absolute()
    file = Path(output_directory).joinpath(output_file_name)
    if os.path.exists(file):
        logging.info("Skipping Download and Extraction")
        return
    os.makedirs(output_directory, exist_ok=True)

    logging.info("Downloading {}.".format(output_file_name))
    download_file_from_google_drive(id, output_directory, output_file_name)

    if file_format is not None and file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Extracting {}.".format(output_file_name))
        extract_archive(file.as_posix())
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Datasets downloaded in {}".format(file))

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
import time
import urllib.error
from pathlib import Path

from torch.hub import download_url_to_file
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, extract_archive

# Errors that indicate a transient/recoverable download failure and are worth retrying.
# ``OSError`` covers socket/timeout/IO errors (``IOError`` is an alias), ``urllib.error.URLError``
# covers HTTP/URL failures, and ``RuntimeError`` is what torchvision raises on a failed download or
# checksum mismatch. Programming errors (e.g. ``TypeError``, ``ValueError``) are deliberately not
# caught here so they surface immediately instead of being retried and masked.
_DOWNLOAD_ERRORS = (OSError, RuntimeError, urllib.error.URLError)


def _remove_partial_files(paths):
    """Delete any files left behind by a failed/partial download attempt.

    Args:
        paths (Iterable[Path]): Paths to remove if they exist. Missing paths are ignored.
    """
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logging.warning("Removed partial download: %s", path)
        except OSError as exc:
            logging.warning("Could not remove partial download %s: %s", path, exc)


def _retry_download(download_fn, retries=3, backoff=2, cleanup_paths=None):
    """Execute ``download_fn`` with retry and exponential backoff.

    Any files listed in ``cleanup_paths`` are removed after a failed attempt so that a
    partially written file is not left in place for the next attempt (or on final failure).

    Args:
        download_fn (callable): Zero-argument callable that performs the download.
        retries (int): Maximum number of attempts. Must be >= 1. Defaults to 3.
        backoff (int): Base for exponential back-off in seconds. Must be >= 1. Defaults to 2.
        cleanup_paths (Iterable[str or Path], optional): Target paths to delete after a failed
            attempt. Defaults to None (nothing to clean up).

    Raises:
        ValueError: If ``retries`` < 1 or ``backoff`` < 1.
        OSError, RuntimeError, urllib.error.URLError: Re-raises the last download error when all
            retries are exhausted.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    if backoff < 1:
        raise ValueError(f"backoff must be >= 1, got {backoff}")
    cleanup_paths = [Path(p) for p in cleanup_paths] if cleanup_paths else []
    for attempt in range(retries):
        try:
            download_fn()
            return
        except _DOWNLOAD_ERRORS as exc:
            _remove_partial_files(cleanup_paths)
            if attempt < retries - 1:
                wait = backoff**attempt
                logging.warning(
                    "Download failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


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
    file = output_directory.joinpath(output_file_name)

    if file.exists():
        logging.info("Skipping Download and Extraction")

        return
    output_directory.mkdir(parents=True, exist_ok=True)

    if file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Downloading and extracting {}.".format(output_file_name))

        _retry_download(
            lambda: download_and_extract_archive(url=url, download_root=output_directory, filename=output_file_name),
            cleanup_paths=[file],
        )
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Downloading {}.".format(output_file_name))
        _retry_download(lambda: download_url_to_file(url, str(file)), cleanup_paths=[file])
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
    file = output_directory.joinpath(output_file_name)
    if file.exists():
        logging.info("Skipping Download and Extraction")
        return
    output_directory.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading {}.".format(output_file_name))
    download_file_from_google_drive(id, output_directory, output_file_name)

    if file_format is not None and file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Extracting {}.".format(output_file_name))
        extract_archive(file.as_posix())
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Datasets downloaded in {}".format(file))

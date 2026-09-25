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
from functools import partial
from pathlib import Path

import pooch
from pooch.hashes import hash_matches
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

_ARCHIVE_FORMATS = ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]

# Errors that indicate a transient/recoverable download failure and are worth retrying.
# ``OSError`` covers socket/timeout/IO errors, ``urllib.error.URLError`` covers HTTP/URL failures,
# and ``RuntimeError`` is what :func:`_retrieve` raises when pooch reports a checksum mismatch.
# Programming errors (e.g. ``TypeError``) are deliberately not caught, so they surface immediately
# instead of being retried and masked.
_DOWNLOAD_ERRORS = (OSError, RuntimeError, urllib.error.URLError)


def _known_hash(md5=None, sha256=None):
    """Build pooch's ``known_hash`` string from the requested checksums.

    ``pooch`` accepts a single ``"<algorithm>:<digest>"`` expectation, so when both are supplied
    the stronger SHA-256 digest is used.

    Args:
        md5 (str, optional): Expected MD5 hex digest. Defaults to None.
        sha256 (str, optional): Expected SHA-256 hex digest. Defaults to None.

    Returns:
        str or None: The ``known_hash`` argument for pooch, or None when no checksum was given.
    """
    if sha256 is not None:
        return f"sha256:{sha256}"
    if md5 is not None:
        return f"md5:{md5}"
    return None


def _hash_ok(file, known_hash):
    """Check ``file`` against ``known_hash``, deleting it if it does not match.

    A file that fails verification is removed so the next download attempt starts clean, and so
    that :func:`torchvision.datasets.utils.download_file_from_google_drive` does not treat the
    corrupt file as already present and skip re-fetching it.

    Args:
        file (Path): Path to the file to check.
        known_hash (str or None): Checksum expectation as built by :func:`_known_hash`. When None,
            there is nothing to check and the file is accepted.

    Returns:
        bool: True if there was nothing to check or the digest matched, False if the file was
        removed as invalid.
    """
    if known_hash is None:
        return True
    if hash_matches(str(file), known_hash):
        return True
    actual = pooch.file_hash(str(file), alg=known_hash.split(":")[0])
    logging.warning("%s does not match %s (got %s); removing it", file, known_hash, actual)
    file.unlink(missing_ok=True)
    return False


def _retrieve(url, known_hash, output_file_name, output_directory):
    """Download ``url`` via pooch, translating a checksum mismatch into a retryable error.

    pooch verifies the checksum, deletes a mismatching download, and reuses an already-valid
    file without re-fetching it. It signals a mismatch with ``ValueError``, which is re-raised as
    ``RuntimeError`` so that :func:`_retry_download` treats a corrupted transfer like any other
    transient failure.

    Args:
        url (str): URL of the object to download.
        known_hash (str or None): Checksum expectation, as built by :func:`_known_hash`.
        output_file_name (str): File name to save the object as.
        output_directory (str or Path): Directory to download into.

    Raises:
        RuntimeError: If the downloaded file does not match ``known_hash``.
    """
    try:
        pooch.retrieve(url, known_hash=known_hash, fname=output_file_name, path=str(output_directory))
    except ValueError as error:
        if "does not match" not in str(error):
            raise
        raise RuntimeError(str(error)) from error


def _retry_download(download_fn, retries=3, backoff=2):
    """Execute ``download_fn`` with retry and exponential backoff.

    Args:
        download_fn (callable): Zero-argument callable that performs the download.
        retries (int): Maximum number of attempts. Must be >= 1. Defaults to 3.
        backoff (int): Base for exponential back-off in seconds. Must be >= 1. Defaults to 2.

    Raises:
        ValueError: If ``retries`` < 1 or ``backoff`` < 1.
        OSError, RuntimeError, urllib.error.URLError: Re-raises the last download error when all
            retries are exhausted.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    if backoff < 1:
        raise ValueError(f"backoff must be >= 1, got {backoff}")
    for attempt in range(retries):
        try:
            download_fn()
            return
        except _DOWNLOAD_ERRORS as exc:
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


def download_file_by_url(url, output_directory, output_file_name, file_format=None, md5=None, sha256=None):
    """Download file/compressed file by url.

    Args:
        url (string): URL of the object to download
        output_directory (string, optional): Full path where object will be saved
                                             Abosolute path recommended. Relative path also works.
        output_file_name (string, optional): File name which object will be saved as
        file_format (string, optional): File format
                                For compressed file, support ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]
        md5 (string, optional): Expected MD5 hex digest of the downloaded file. When provided, the download
                                is verified and a mismatch is retried, then raised. Defaults to None.
        sha256 (string, optional): Expected SHA-256 hex digest of the downloaded file. Takes precedence
                                over ``md5`` when both are given. Defaults to None.

    Raises:
        RuntimeError: If verification is requested and the downloaded file does not match after all retries.

    Example: (Grab the raw link from GitHub. Notice that using "raw" in the URL.)
        >>> url = "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_04_train.pkl"
        >>> download_file_by_url(url, "data", "a.pkl", "pkl")

        >>> url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
        >>> download_file_by_url(url, "data", "video_test_data.zip", "zip")

        >>> url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"
        >>> download_file_by_url(url, "data", "video_test_data.zip", "zip", md5="0123...")

    """

    output_directory = Path(output_directory).absolute()
    file = output_directory.joinpath(output_file_name)
    known_hash = _known_hash(md5, sha256)

    if file.exists() and known_hash is None:
        # Historical fast path: with no checksum to check against, an existing file is reused as-is.
        logging.info("Skipping Download and Extraction")
        return
    output_directory.mkdir(parents=True, exist_ok=True)

    # When a checksum is given, pooch verifies any existing file and re-downloads it on mismatch,
    # so a corrupt cached file is never silently reused.
    logging.info("Downloading {}.".format(output_file_name))
    _retry_download(partial(_retrieve, url, known_hash, output_file_name, output_directory))

    if file_format in _ARCHIVE_FORMATS:
        logging.info("Extracting {}.".format(output_file_name))
        extract_archive(str(file), str(output_directory))
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Datasets downloaded in {}".format(file))


def _fetch_gdrive(id, output_directory, output_file_name, file, known_hash):
    """Download one Google Drive file and verify it, for use inside :func:`_retry_download`.

    Args:
        id (str): Google Drive file id of the object to download.
        output_directory (Path): Directory to download into.
        output_file_name (str): File name to save the object as.
        file (Path): Full path to the downloaded file.
        known_hash (str or None): Checksum expectation as built by :func:`_known_hash`.

    Raises:
        RuntimeError: If the downloaded file does not match ``known_hash``.
    """
    download_file_from_google_drive(id, output_directory, output_file_name)
    if not _hash_ok(file, known_hash):
        raise RuntimeError(f"{file} does not match the expected {known_hash}. Deleted download for safety.")


def download_file_gdrive(id, output_directory, output_file_name, file_format=None, md5=None, sha256=None):
    """Download file/compressed file by Google Drive id.

    Args:
        id (string): Google Drive file id of the object to download
        output_directory (string, optional): Full path where object will be saved
                                             Abosolute path recommended. Relative path also works.
        output_file_name (string, optional): File name which object will be saved as
        file_format (string, optional): File format
                                For compressed file, support ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]
        md5 (string, optional): Expected MD5 hex digest of the downloaded file. When provided, the download
                                is verified and a mismatch is retried, then raised. Defaults to None.
        sha256 (string, optional): Expected SHA-256 hex digest of the downloaded file. Takes precedence
                                over ``md5`` when both are given. Defaults to None.

    Raises:
        RuntimeError: If verification is requested and the downloaded file does not match after all retries.

    Example:
        >>> gdrive_id = "1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga"
        >>> download_file_gdrive(gdrive_id, "data", "demo_datasets.zip", "zip")

        >>> gdrive_id = "1SV7fmAnWj-6AU9X5BGOrvGMoh2Gu9Nih"
        >>> download_file_gdrive(gdrive_id, "data", "dummy_data.csv", "csv")
    """

    output_directory = Path(output_directory).absolute()
    file = output_directory.joinpath(output_file_name)
    known_hash = _known_hash(md5, sha256)

    # An existing file is reused only if it passes verification; _hash_ok removes it otherwise, so
    # a corrupt cached file is re-downloaded rather than silently accepted.
    if file.exists() and _hash_ok(file, known_hash):
        logging.info("Skipping Download and Extraction")
        return
    output_directory.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading {}.".format(output_file_name))
    # pooch cannot download from Google Drive, so torchvision fetches and pooch's hashing verifies.
    _retry_download(partial(_fetch_gdrive, id, output_directory, output_file_name, file, known_hash))

    if file_format is not None and file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Extracting {}.".format(output_file_name))
        extract_archive(file.as_posix())
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Datasets downloaded in {}".format(file))

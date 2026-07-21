# ===============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Raivo Koot, rekoot1@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# ===============================================================================

"""Data downloading and compressed data extraction functions, Based on
https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py
https://github.com/pytorch/pytorch/blob/master/torch/hub.py
"""

import hashlib
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


def _hash_file(file, algorithm, chunk_size=1024 * 1024):
    """Compute the hex digest of a file, reading it in chunks.

    Args:
        file (str or Path): Path to the file to hash.
        algorithm (str): Hash algorithm name understood by :func:`hashlib.new` (e.g. "md5", "sha256").
        chunk_size (int): Number of bytes read per iteration. Defaults to 1 MiB.

    Returns:
        str: The lowercase hexadecimal digest.
    """
    hasher = hashlib.new(algorithm)
    with open(file, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_file(file, md5=None, sha256=None, file_size=None):
    """Verify a downloaded file against optional checksums and/or an expected size.

    Verification is skipped entirely when no expectation is provided. A mismatch raises
    ``RuntimeError`` (rather than ``ValueError``) so that, when called from within
    :func:`_retry_download`, a corrupted download is retried and cleaned up like any other
    transient download failure.

    Args:
        file (str or Path): Path to the downloaded file.
        md5 (str, optional): Expected MD5 hex digest. Defaults to None (not checked).
        sha256 (str, optional): Expected SHA-256 hex digest. Defaults to None (not checked).
        file_size (int, optional): Expected size in bytes. Defaults to None (not checked).

    Raises:
        RuntimeError: If the file is missing or any provided expectation does not match.
    """
    if md5 is None and sha256 is None and file_size is None:
        return
    file = Path(file)
    if not file.exists():
        raise RuntimeError(f"Cannot verify {file}: file does not exist")
    if file_size is not None:
        actual_size = file.stat().st_size
        if actual_size != file_size:
            raise RuntimeError(f"Size mismatch for {file}: expected {file_size} bytes, got {actual_size}")
    for algorithm, expected in (("md5", md5), ("sha256", sha256)):
        if expected is None:
            continue
        actual = _hash_file(file, algorithm)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{algorithm} mismatch for {file}: expected {expected}, got {actual}")
    logging.info("Verified integrity of %s", file)


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


def download_file_by_url(
    url, output_directory, output_file_name, file_format=None, md5=None, sha256=None, file_size=None
):
    """Download file/compressed file by url.

    Args:
        url (string): URL of the object to download
        output_directory (string, optional): Full path where object will be saved
                                             Abosolute path recommended. Relative path also works.
        output_file_name (string, optional): File name which object will be saved as
        file_format (string, optional): File format
                                For compressed file, support ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]
        md5 (string, optional): Expected MD5 hex digest of the downloaded file. When provided, the
                                download is verified and a mismatch is retried, then raised. Defaults to None.
        sha256 (string, optional): Expected SHA-256 hex digest of the downloaded file. Defaults to None.
        file_size (int, optional): Expected size of the downloaded file in bytes. Defaults to None.

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

    if file.exists():
        logging.info("Skipping Download and Extraction")

        return
    output_directory.mkdir(parents=True, exist_ok=True)

    if file_format in ["tar.xz", "tar", "tar.gz", "tgz", "gz", "zip"]:
        logging.info("Downloading and extracting {}.".format(output_file_name))

        def _download_and_extract():
            download_and_extract_archive(url=url, download_root=output_directory, filename=output_file_name)
            _verify_file(file, md5=md5, sha256=sha256, file_size=file_size)

        _retry_download(_download_and_extract, cleanup_paths=[file])
        logging.info("Datasets downloaded and extracted in {}".format(file))
    else:
        logging.info("Downloading {}.".format(output_file_name))

        def _download():
            download_url_to_file(url, str(file))
            _verify_file(file, md5=md5, sha256=sha256, file_size=file_size)

        _retry_download(_download, cleanup_paths=[file])
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

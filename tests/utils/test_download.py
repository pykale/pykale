import hashlib
import os
from pathlib import Path
from unittest.mock import call, MagicMock, patch

import pytest

from kale.utils.download import _known_hash, _retrieve, _retry_download, download_file_by_url, download_file_gdrive

output_directory = Path().absolute().joinpath("tests/test_data/download")
PARAM = [
    "https://github.com/pykale/data/raw/main/videos/video_test_data/ADL/annotations/labels_train_test/adl_P_11_train.pkl;a.pkl;pkl",
    "https://github.com/pykale/data/raw/main/videos/video_test_data.zip;video_test_data.zip;zip",
]

GDRIVE_PARAM = [
    "1U4D23R8u8MJX9KVKb92bZZX-tbpKWtga;demo_datasets.zip;zip",
    "1SV7fmAnWj-6AU9X5BGOrvGMoh2Gu9Nih;dummy_data.csv;csv",
]

# pooch reports a checksum mismatch with this wording; ``_retrieve`` keys off it to tell a retryable
# corrupt download apart from a genuine programming error.
POOCH_MISMATCH = (
    "MD5 hash of downloaded file (data.pkl) does not match the known hash: "
    "expected md5:000 but got abc. Deleted download for safety."
)


def test_retry_download_succeeds_on_first_attempt():
    fn = MagicMock()
    _retry_download(fn, retries=3, backoff=2)
    fn.assert_called_once()


def test_retry_download_retries_on_failure():
    fn = MagicMock(side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), None])
    with patch("kale.utils.download.time.sleep") as mock_sleep:
        _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3
    mock_sleep.assert_has_calls([call(1), call(2)])


def test_retry_download_raises_after_all_retries():
    fn = MagicMock(side_effect=RuntimeError("timeout"))
    with patch("kale.utils.download.time.sleep"):
        with pytest.raises(RuntimeError, match="timeout"):
            _retry_download(fn, retries=3, backoff=2)
    assert fn.call_count == 3


@pytest.mark.parametrize("kwargs", [{"retries": 0}, {"retries": -1}, {"backoff": 0}, {"backoff": -1}])
def test_retry_download_invalid_args(kwargs):
    with pytest.raises(ValueError):
        _retry_download(MagicMock(), **kwargs)


def test_retry_download_does_not_retry_programming_errors():
    # A non-download error (e.g. TypeError) is not one of _DOWNLOAD_ERRORS, so it must
    # propagate immediately without being retried and masked.
    fn = MagicMock(side_effect=TypeError("bad call"))
    with patch("kale.utils.download.time.sleep") as mock_sleep:
        with pytest.raises(TypeError, match="bad call"):
            _retry_download(fn, retries=3, backoff=2)
    fn.assert_called_once()
    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    "md5, sha256, expected",
    [
        (None, None, None),
        ("abc", None, "md5:abc"),
        (None, "def", "sha256:def"),
        ("abc", "def", "sha256:def"),  # sha256 wins when both are supplied
    ],
)
def test_known_hash(md5, sha256, expected):
    assert _known_hash(md5, sha256) == expected


def test_retrieve_translates_checksum_mismatch_to_runtime_error(tmp_path):
    # pooch signals a checksum mismatch with ValueError. `_retrieve` re-raises it as RuntimeError
    # so that _retry_download treats a corrupt transfer as retryable.
    with patch("kale.utils.download.pooch.retrieve", side_effect=ValueError(POOCH_MISMATCH)):
        with pytest.raises(RuntimeError, match="does not match"):
            _retrieve("http://example.com/data.pkl", "md5:000", "data.pkl", tmp_path)


def test_retrieve_propagates_other_value_errors(tmp_path):
    # A ValueError that is not a checksum mismatch (e.g. an unsupported URL protocol) is a
    # programming/config error and must surface immediately rather than being retried.
    with patch("kale.utils.download.pooch.retrieve", side_effect=ValueError("Unrecognized URL protocol")):
        with pytest.raises(ValueError, match="Unrecognized URL protocol"):
            _retrieve("ftp://example.com/data.pkl", None, "data.pkl", tmp_path)


def test_download_file_by_url_plain_calls_pooch(tmp_path):
    with patch("kale.utils.download.pooch.retrieve") as mock_retrieve:
        download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl")
    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args.kwargs["known_hash"] is None


def test_download_file_by_url_archive_extracts(tmp_path):
    with patch("kale.utils.download.pooch.retrieve") as mock_retrieve:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_by_url("http://example.com/data.zip", tmp_path, "data.zip", "zip")
    mock_retrieve.assert_called_once()
    mock_extract.assert_called_once()


def test_download_file_by_url_passes_checksum_to_pooch(tmp_path):
    with patch("kale.utils.download.pooch.retrieve") as mock_retrieve:
        download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl", md5="0" * 32)
    assert mock_retrieve.call_args.kwargs["known_hash"] == "md5:" + "0" * 32


def test_download_file_by_url_skips_existing_file_without_checksum(tmp_path):
    # With nothing to verify, an existing file is reused without touching the network.
    (tmp_path / "data.pkl").write_bytes(b"already here")
    with patch("kale.utils.download.pooch.retrieve") as mock_retrieve:
        download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl")
    mock_retrieve.assert_not_called()


def test_download_file_by_url_verifies_existing_file_when_checksum_given(tmp_path):
    # Regression guard: a cached file must NOT bypass verification when a checksum is supplied,
    # otherwise a corrupt cached file would be silently reused.
    content = b"already here"
    (tmp_path / "data.pkl").write_bytes(content)
    with patch("kale.utils.download.pooch.retrieve") as mock_retrieve:
        download_file_by_url(
            "http://example.com/data.pkl", tmp_path, "data.pkl", "pkl", md5=hashlib.md5(content).hexdigest()
        )
    mock_retrieve.assert_called_once()


def test_download_file_by_url_checksum_mismatch_retries_then_raises(tmp_path):
    # Every attempt fails verification, so the error surfaces once retries are exhausted.
    with patch("kale.utils.download.time.sleep"):
        with patch("kale.utils.download.pooch.retrieve", side_effect=ValueError(POOCH_MISMATCH)) as mock_retrieve:
            with pytest.raises(RuntimeError, match="does not match"):
                download_file_by_url("http://example.com/data.pkl", tmp_path, "data.pkl", "pkl", md5="0" * 32)
    assert mock_retrieve.call_count == 3


def test_download_file_by_url_archive_not_extracted_on_failure(tmp_path):
    # A download that never verifies must not reach extraction.
    with patch("kale.utils.download.time.sleep"):
        with patch("kale.utils.download.pooch.retrieve", side_effect=ValueError(POOCH_MISMATCH)):
            with patch("kale.utils.download.extract_archive") as mock_extract:
                with pytest.raises(RuntimeError):
                    download_file_by_url("http://example.com/data.zip", tmp_path, "data.zip", "zip", md5="0" * 32)
    mock_extract.assert_not_called()


@pytest.mark.parametrize("param", PARAM)
def test_download_file_by_url(param):
    url, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_by_url(url, output_directory, output_file_name, file_format)
    download_file_by_url(url, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()


def test_download_file_gdrive_archive_mocked(tmp_path):
    # Exercise the gdrive download + extract branch without hitting the network.
    def fake_gdrive(id, root, name):
        Path(root).joinpath(name).write_bytes(b"archive-bytes")

    with patch("kale.utils.download.download_file_from_google_drive", side_effect=fake_gdrive) as mock_dl:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_gdrive("some-id", tmp_path, "data.zip", "zip")
    mock_dl.assert_called_once()
    mock_extract.assert_called_once()
    assert (tmp_path / "data.zip").exists()


def test_download_file_gdrive_plain_mocked(tmp_path):
    # Exercise the gdrive plain (no-extract) branch without hitting the network.
    def fake_gdrive(id, root, name):
        Path(root).joinpath(name).write_bytes(b"plain-bytes")

    with patch("kale.utils.download.download_file_from_google_drive", side_effect=fake_gdrive) as mock_dl:
        with patch("kale.utils.download.extract_archive") as mock_extract:
            download_file_gdrive("some-id", tmp_path, "data.csv", "csv")
    mock_dl.assert_called_once()
    mock_extract.assert_not_called()
    assert (tmp_path / "data.csv").exists()


@pytest.mark.parametrize("param", GDRIVE_PARAM)
def test_download_file_gdrive(param):
    id, output_file_name, file_format = param.split(";")

    # run twice to test the code when the file exist
    download_file_gdrive(id, output_directory, output_file_name, file_format)
    download_file_gdrive(id, output_directory, output_file_name, file_format)

    assert os.path.exists(output_directory.joinpath(output_file_name)) is True
    assert output_directory.exists()

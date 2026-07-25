import pandas as pd
import pytest

from kale.loaddata.video_datasets import BasicVideoDataset


def _make_annotation_file(tmp_path, rows):
    """Write an annotation DataFrame to a pickle, mirroring the EPIC-Kitchen list format.

    ``make_dataset`` reads columns 0 (video name), 1 (start frame), 2 (end frame) and 5 (label).
    """
    frame = pd.DataFrame(rows, columns=["video", "start", "end", "spare_a", "spare_b", "label"])
    path = tmp_path / "annotations.pkl"
    frame.to_pickle(path)
    return path


def _dataset_for(annotation_path, n_classes=8):
    """Build a BasicVideoDataset without running __init__, which needs real image folders."""
    dataset = object.__new__(BasicVideoDataset)
    dataset.annotationfile_path = str(annotation_path)
    dataset.n_classes = n_classes
    dataset.dataset = "train"
    return dataset


class TestBasicVideoDatasetMakeDataset:
    """Tests for ``BasicVideoDataset.make_dataset`` annotation parsing (issue #556)."""

    def test_parses_annotation_fields_as_integers(self, tmp_path):
        """Start frame, end frame and label are parsed to ints, and out-of-range labels are dropped."""
        path = _make_annotation_file(
            tmp_path,
            [
                ["video_1", "10", "20", "", "", "3"],
                ["video_2", "30", "45", "", "", "0"],
                ["video_3", "50", "60", "", "", "9"],  # label >= n_classes, excluded
            ],
        )

        data = _dataset_for(path, n_classes=8).make_dataset()

        assert data == [("video_1", 10, 20, 3), ("video_2", 30, 45, 0)]
        for _, start, end, label in data:
            assert isinstance(start, int) and isinstance(end, int) and isinstance(label, int)

    def test_malicious_annotation_value_is_not_executed(self, tmp_path):
        """A code-like annotation value is rejected rather than evaluated.

        ``make_dataset`` previously ran ``eval`` on these fields, so a crafted pickle could execute
        arbitrary code. Parsing with ``int`` must raise instead, and must leave no side effect.
        """
        marker = tmp_path / "marker.txt"
        payload = f"__import__('pathlib').Path(r'{marker}').write_text('executed')"
        path = _make_annotation_file(tmp_path, [["video_1", "10", "20", "", "", payload]])

        with pytest.raises(ValueError):
            _dataset_for(path).make_dataset()

        assert not marker.exists(), "annotation value was executed instead of parsed"

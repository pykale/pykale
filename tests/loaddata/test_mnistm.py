import os

import pytest
import torch

from kale.loaddata.mnistm import MNISTM


def _write_processed_files(root):
    """Create MNIST-M processed .pt files the same way MNISTM.download() saves them.

    Each file holds a plain ``(ByteTensor image data, label tensor)`` tuple, so it must load
    under ``weights_only=True`` without any custom classes.
    """
    processed = os.path.join(root, "MNISTM", "processed")
    os.makedirs(processed)
    for name in (MNISTM.training_file, MNISTM.test_file):
        images = torch.randint(0, 256, (4, 28, 28, 3), dtype=torch.uint8)
        labels = torch.randint(0, 10, (4,))
        torch.save((images, labels), os.path.join(processed, name))


@pytest.mark.parametrize("train", [True, False])
def test_mnistm_loads_with_weights_only(tmp_path, train):
    # Regression test for the weights_only=True switch: the processed tuple must still load,
    # and a sample must be retrievable (as exercised by the digits_dann example).
    _write_processed_files(str(tmp_path))

    dataset = MNISTM(root=str(tmp_path), train=train, download=False)

    assert len(dataset) == 4
    assert dataset.data.dtype == torch.uint8
    image, target = dataset[0]
    assert image.size == (28, 28)  # PIL Image (W, H)

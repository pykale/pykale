import os

import matplotlib.figure
import numpy as np
import pytest
from numpy import testing

from kale.interpret.visualize import plot_multi_images
from kale.loaddata.image_access import dicom2array, read_dicom_dir
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url

SCALES = [4, 8]
cmr_url = "https://github.com/pykale/data/raw/main/images/ShefPAH-179/SA_64x64.zip"


@pytest.fixture(scope="module")
def images(download_path):
    download_file_by_url(cmr_url, download_path, "SA_64x64.zip", "zip")
    img_path = os.path.join(download_path, "SA_64x64", "DICOM")
    cmr_ds = read_dicom_dir(img_path, sort_instance=True, sort_patient=True)
    cmr_images = dicom2array(dicom_ds=cmr_ds, return_ids=False)

    return cmr_images[:5, ...]


@pytest.fixture(scope="module")
def coords():
    landmarks = np.asarray(
        [
            [32.0, 39.75, 29.25, 23.75, 19.0, 41.0],
            [24.5, 40.0, 28.5, 23.75, 11.0, 37.25],
            [26.25, 40.5, 27.75, 24.25, 12.25, 40.75],
            [34.25, 38.0, 34.25, 21.25, 23.0, 41.0],
            [33.0, 40.25, 31.5, 24.25, 19.5, 40.5],
        ]
    )
    return landmarks


def test_reg(images, coords):
    marker_kwargs = {"marker": "o", "markerfacecolor": (1, 1, 1, 0.1), "markeredgewidth": 1.5, "markeredgecolor": "r"}
    im_kwargs = {"cmap": "gray"}
    marker_names = ["inf insertion point", "sup insertion point", "RV inf"]

    n_samples = images.shape[0]
    fig = plot_multi_images(
        images[:, 0, ...],
        n_cols=5,
        marker_locs=coords,
        marker_titles=marker_names,
        marker_cmap="Set1",
        im_kwargs=im_kwargs,
        marker_kwargs=marker_kwargs,
    )
    assert type(fig) == matplotlib.figure.Figure
    with pytest.raises(Exception):
        reg_img_stack(images, coords[1:, :])
    images_reg, max_dist = reg_img_stack(images, coords)
    # images after registration should be close to original images, because values of noise are small
    testing.assert_allclose(images_reg, images)
    # add one for avoiding inf relative difference
    testing.assert_allclose(max_dist + 1, np.ones(n_samples), rtol=2, atol=2)
    fig = plot_multi_images(images_reg[:, 0, ...], n_cols=5)
    assert type(fig) == matplotlib.figure.Figure


@pytest.mark.parametrize("scale", SCALES)
def test_rescale(scale, images):
    img_rescaled = rescale_img_stack(images, 1 / scale)
    # dim1 and dim2 have been rescaled
    testing.assert_equal(img_rescaled.shape[-1], round(images.shape[-1] / scale))
    testing.assert_equal(img_rescaled.shape[-2], round(images.shape[-2] / scale))
    # n_phases and n_samples are unchanged
    testing.assert_equal(img_rescaled.shape[:2], images.shape[:2])


def test_masking(images):
    # generate synthetic mask randomly
    mask = np.random.randint(0, 2, size=(images.shape[-2], images.shape[-1]))
    idx_zeros = np.where(mask == 0)
    idx_ones = np.where(mask == 1)
    img_masked = mask_img_stack(images, mask)
    n_samples, n_phases = images.shape[:2]
    for i in range(n_samples):
        for j in range(n_phases):
            img = img_masked[i, j, ...]
            testing.assert_equal(np.sum(img[idx_zeros]), 0)
            img_orig = images[i, j, ...]
            testing.assert_equal(img[idx_ones], img_orig[idx_ones])


def test_normalize(images):
    norm_image = normalize_img_stack(images)
    assert np.min(norm_image) >= 0
    assert np.max(norm_image) <= 1

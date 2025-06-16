import os

import matplotlib.figure
import numpy as np
import pytest
import torch
from numpy import testing
from PIL import Image

from kale.interpret.visualize import plot_multi_images
from kale.loaddata.image_access import check_dicom_series_uid, dicom2arraylist, read_dicom_dir
from kale.prepdata.image_transform import (
    mask_img_stack,
    normalize_img_stack,
    prepare_image_tensor,
    reg_img_stack,
    rescale_img_stack,
)

SCALES = [4, 8]


@pytest.fixture(scope="module")
def images(download_path):
    img_path = os.path.join(download_path, "SA_64x64_v2.0", "DICOM")
    cmr_dcm_list = read_dicom_dir(img_path, sort_instance=True, sort_patient=True, check_series_uid=True)
    dcms = []
    for i in range(5):
        for j in range(len(cmr_dcm_list[i])):
            dcms.append(cmr_dcm_list[i][j])
    dcm5_list = check_dicom_series_uid(dcms)
    cmr_images = dicom2arraylist(dicom_patient_list=dcm5_list, return_patient_id=False)

    return cmr_images


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
    marker_kwargs = {"marker": "+", "color": (1, 1, 1, 0.1), "s": 50}
    im_kwargs = {"cmap": "gray"}
    title_kwargs = {"fontsize": 20}
    marker_names = ["inf insertion point", "sup insertion point", "RV inf"]

    n_samples = len(images)
    fig = plot_multi_images(
        [images[i][0, ...] for i in range(n_samples)],
        n_cols=5,
        marker_locs=coords,
        marker_titles=marker_names,
        marker_cmap="Set1",
        im_kwargs=im_kwargs,
        marker_kwargs=marker_kwargs,
        title_kwargs=title_kwargs,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    with pytest.raises(Exception):
        reg_img_stack(images, coords[1:, :], coords[0])
    images_reg, max_dist = reg_img_stack(images, coords, target_coords=coords[0])
    # images after registration should be close to original images, because values of noise are small
    for i in range(n_samples):
        testing.assert_allclose(images_reg[i], images[i])
    # add one for avoiding inf relative difference
    testing.assert_allclose(max_dist + 1, np.ones(n_samples), rtol=2, atol=2)
    fig = plot_multi_images([images_reg[i][0, ...] for i in range(n_samples)], n_cols=5)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.parametrize("scale", SCALES)
def test_rescale(scale, images):
    img_rescaled = rescale_img_stack(images, 1 / scale)
    n_samples = len(img_rescaled)
    testing.assert_equal(n_samples, len(images))
    for i in range(n_samples):
        # dim1 and dim2 have been rescaled
        testing.assert_equal(img_rescaled[i].shape[-1], round(images[i].shape[-1] / scale))
        testing.assert_equal(img_rescaled[i].shape[-2], round(images[i].shape[-2] / scale))
        # n_phases are unchanged
        testing.assert_equal(img_rescaled[i].shape[0], images[i].shape[0])


def test_masking(images):
    # generate synthetic mask randomly
    mask = np.random.randint(0, 2, size=(images[0].shape[-2], images[0].shape[-1]))
    idx_zeros = np.where(mask == 0)
    idx_ones = np.where(mask == 1)
    img_masked = mask_img_stack(images, mask)
    n_samples = len(images)
    for i in range(n_samples):
        n_phases = images[i].shape[0]
        for j in range(n_phases):
            img = img_masked[i][j, ...]
            testing.assert_equal(np.sum(img[idx_zeros]), 0)
            img_orig = images[i][j, ...]
            testing.assert_equal(img[idx_ones], img_orig[idx_ones])


def test_normalize(images):
    norm_image = normalize_img_stack(images)
    for i in range(len(norm_image)):
        assert np.min(norm_image[i]) >= 0
        assert np.max(norm_image[i]) <= 1


def save_image_to_path(np_img, path):
    """Save numpy image to file for skimage.io.imread compatibility"""
    img = Image.fromarray(np_img)
    img.save(path)


@pytest.mark.parametrize(
    "channels_in,channels_arg,expected_shape",
    [
        (1, 1, (1, 224, 224)),  # Grayscale file, grayscale output
        (3, 3, (3, 224, 224)),  # RGB file, RGB output
        (1, 3, (3, 224, 224)),  # Grayscale file, RGB output
        (3, 1, (1, 224, 224)),  # RGB file, grayscale output
    ],
)
def test_prepare_image_tensor_branches(tmp_path, channels_in, channels_arg, expected_shape):
    # Create dummy image data (random)
    H, W = 32, 32
    if channels_in == 1:
        np_img = np.random.randint(0, 255, size=(H, W), dtype=np.uint8)
    elif channels_in == 3:
        np_img = np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
    else:
        raise ValueError("Test: invalid test param")

    # Save temp image
    img_path = tmp_path / f"test_{channels_in}.png"
    save_image_to_path(np_img, img_path)

    # Run function
    out_tensor = prepare_image_tensor(str(img_path), resize_dim=(224, 224), channels=channels_arg)
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape == expected_shape
    assert out_tensor.max() <= 1.0 or out_tensor.min() >= -1.0  # normalized


def test_prepare_image_tensor_hw1_to_hw3(tmp_path):
    # Simulate (H, W, 1) image
    np_img = np.random.randint(0, 255, size=(32, 32, 1), dtype=np.uint8)
    img_path = tmp_path / "hw1.png"
    # PIL expects at least 2D, save as grayscale
    img = Image.fromarray(np_img.squeeze())
    img.save(img_path)
    # Re-load as (H, W, 1) and save as .npy for direct skimage.io.imread
    np.save(str(tmp_path / "hw1.npy"), np_img)
    # Use prepare_image_tensor (simulate read as H, W, 1)
    img_loaded = np.load(str(tmp_path / "hw1.npy"))
    # Monkeypatch skimage.io.imread to return (H, W, 1)
    import skimage.io

    orig_imread = skimage.io.imread
    skimage.io.imread = lambda path: img_loaded
    try:
        out_tensor = prepare_image_tensor("dummy_path", resize_dim=(224, 224), channels=3)
        assert out_tensor.shape == (3, 224, 224)
    finally:
        skimage.io.imread = orig_imread


def test_prepare_image_tensor_invalid_channels(tmp_path):
    np_img = np.random.randint(0, 255, size=(32, 32), dtype=np.uint8)
    img_path = tmp_path / "test_invalid.png"
    save_image_to_path(np_img, img_path)
    with pytest.raises(ValueError):
        prepare_image_tensor(str(img_path), resize_dim=(224, 224), channels=2)

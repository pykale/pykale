import numpy as np
import pytest
from numpy import testing

from kale.prepdata.image_transform import mask_img_stack, reg_img_stack, rescale_img_stack

SCALES = [4, 8]


@pytest.fixture(scope="module")
def images(gait):
    return gait["fea3D"][..., :10]


def test_reg(images):
    n_samples = images.shape[-1]
    # generate synthetic coordinates
    coords = np.ones((n_samples, 4))
    coords[:, 2:] += 20
    # use first row as destination coordinates, add small random noise to the remaining coordinates
    coords[1:, :] += np.random.random(size=(n_samples - 1, 4))
    with pytest.raises(Exception):
        reg_img_stack(images, coords[1:, :])
    images_reg, max_dist = reg_img_stack(images, coords)
    # images after registration should be close to original images, because values of noise are small
    testing.assert_allclose(images_reg, images)
    # add one for avoiding inf relative difference
    testing.assert_allclose(max_dist + 1, np.ones(n_samples))


@pytest.mark.parametrize("scale", SCALES)
def test_rescale(scale, images):
    img_rescaled = rescale_img_stack(images, scale)
    # dim1 and dim2 have been rescaled
    testing.assert_equal(img_rescaled.shape[0], round(images.shape[0] / scale))
    testing.assert_equal(img_rescaled.shape[1], round(images.shape[1] / scale))
    # n_phases and n_samples are unchanged
    testing.assert_equal(img_rescaled.shape[-2:], images.shape[-2:])


def test_masking(images):
    # generate synthetic mask randomly
    mask = np.random.randint(0, 2, size=(images.shape[0], images.shape[1]))
    idx_zeros = np.where(mask == 0)
    idx_ones = np.where(mask == 1)
    img_masked = mask_img_stack(images, mask)
    n_phases, n_samples = images.shape[-2:]
    for i in range(n_phases):
        for j in range(n_phases):
            img = img_masked[..., j, i]
            testing.assert_equal(np.sum(img[idx_zeros]), 0)
            testing.assert_equal(img_masked[idx_ones], images[idx_ones])

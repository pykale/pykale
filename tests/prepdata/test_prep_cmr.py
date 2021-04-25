import numpy as np
import pytest

from numpy import testing
from scipy.io import loadmat

from kale.prepdata.prep_cmr import reg_img_stack, rescale_img_stack, mask_img_stack

gait = loadmat("../test_data/gait_gallery_data.mat")
# images = gait["fea3D"].transpose((3, 0, 1, 2))
images = gait["fea3D"][..., :10]

SCALES = [4, 8]


def test_reg():
    n_samples = images.shape[-1]
    coords = np.ones((n_samples, 2))
    coords[1:, :] += np.random.random(size=(n_samples - 1, 2))
    images_reg, max_dist = reg_img_stack(images, coords)
    testing.assert_equal(images_reg.shape, images.shape)
    testing.assert_equal(max_dist.shape, (n_samples,))


@pytest.mark.parametrize("scale", SCALES)
def test_rescale(scale):
    img_rescaled = rescale_img_stack(images, scale)
    testing.assert_equal(img_rescaled.shape[0], round(images.shape[0] / scale))
    testing.assert_equal(img_rescaled.shape[1], round(images.shape[1] / scale))


def test_masking():
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

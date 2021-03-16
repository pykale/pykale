import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat
from tensorly.tenalg import multi_mode_dot

from kale.embed.mpca import MPCA

gait = loadmat("tests/test_data/gait_gallery_data.mat")
baseline_model = loadmat("tests/test_data/mpca_baseline_res.mat")

N_COMPS = [1, 5, 20, 50, 100]
VAR_RATIOS = [0.85, 0.9, 0.95]


@pytest.mark.parametrize("n_components", N_COMPS)
@pytest.mark.parametrize("var_ratio", VAR_RATIOS)
def test_mpca(var_ratio, n_components):
    # basic mpca test, return tensor
    x = gait["fea3D"].transpose((3, 0, 1, 2))
    mpca = MPCA(var_ratio=var_ratio, return_vector=False)
    x_proj = mpca.fit(x).transform(x)

    testing.assert_equal(x_proj.ndim, x.ndim)
    testing.assert_equal(x_proj.shape[0], x.shape[0])
    assert mpca.n_components <= np.prod(x.shape[1:])
    assert n_components < mpca.n_components
    for i in range(1, x.ndim):
        assert x_proj.shape[i] <= x.shape[i]
        testing.assert_equal(mpca.proj_mats[i - 1].shape[1], x.shape[i])

    x_rec = mpca.inverse_transform(x_proj)
    testing.assert_equal(x_rec.shape, x.shape)

    # test return vector
    mpca.set_params(**{"return_vector": True, "n_components": n_components})

    x_proj = mpca.transform(x)
    testing.assert_equal(x_proj.ndim, 2)
    testing.assert_equal(x_proj.shape[0], x.shape[0])
    testing.assert_equal(x_proj.shape[1], n_components)
    x_rec = mpca.inverse_transform(x_proj)
    testing.assert_equal(x_rec.shape, x.shape)

    # test n_samples = 1
    x0_proj = mpca.transform(x[0])
    testing.assert_equal(x0_proj.ndim, 2)
    testing.assert_equal(x0_proj.shape[0], 1)
    testing.assert_equal(x0_proj.shape[1], n_components)
    x0_rec = mpca.inverse_transform(x0_proj)
    testing.assert_equal(x0_rec.shape[1:], x[0].shape)

    # test n_components exceeds upper limit
    mpca.set_params(**{"return_vector": True, "n_components": np.prod(x.shape[1:]) + 1})
    x_proj = mpca.transform(x)
    testing.assert_equal(x_proj.shape[1], np.prod(mpca.shape_out))


def test_mpca_against_baseline():
    x = gait["fea3D"].transpose((3, 0, 1, 2))
    baseline_proj_mats = [baseline_model["tUs"][i][0] for i in range(baseline_model["tUs"].size)]
    mpca = MPCA(var_ratio=0.97)
    x_proj = mpca.fit(x).transform(x)
    baseline_proj_x = multi_mode_dot(x, baseline_proj_mats, modes=[1, 2, 3])
    # check whether the output shape is consistent with the baseline output by keeping the same variance ratio 97%
    testing.assert_equal(x_proj.shape, baseline_proj_x.shape)

    for i in range(x.ndim - 1):
        # check whether each eigen-vector column is equal to/opposite of corresponding baseline eigen-vector column
        for j in range(baseline_proj_mats[i].shape[0]):
            # subtraction of eigen-vector columns
            eig_col_sub = mpca.proj_mats[i][j:] - baseline_proj_mats[i][j:]
            # sum of eigen-vector columns
            eig_col_sum = mpca.proj_mats[i][j:] + baseline_proj_mats[i][j:]
            compare_ = np.multiply(eig_col_sub, eig_col_sum)
            # plus one for avoiding inf relative difference
            testing.assert_allclose(compare_ + 1, np.ones(compare_.shape))

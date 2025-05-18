import os

import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from tensorly.tenalg import multi_mode_dot

from kale.embed.factorization import MIDA, MPCA

from ..helpers.toy_dataset import make_domain_shifted_dataset

N_COMPS = [1, 50, 100]
VAR_RATIOS = [0.7, 0.95]
relative_tol = 0.00001
baseline_url = "https://github.com/pykale/data/raw/main/videos/gait/mpca_baseline.mat"


@pytest.fixture(scope="module")
def baseline_model(download_path):
    return loadmat(os.path.join(download_path, "baseline.mat"))


@pytest.fixture(scope="module")
def sample_data():
    # Test an extreme case of domain shift
    # yet the data's manifold is linearly separable
    x, y, domains = make_domain_shifted_dataset(
        num_domains=10,
        num_samples_per_class=2,
        num_features=20,
        centroid_shift_scale=32768,
        random_state=0,
    )

    factors = OneHotEncoder(handle_unknown="ignore").fit_transform(domains.reshape(-1, 1)).toarray()

    return x, y, domains, factors


@pytest.mark.parametrize("n_components", N_COMPS)
@pytest.mark.parametrize("var_ratio", VAR_RATIOS)
def test_mpca(var_ratio, n_components, gait):
    # basic mpca test, return tensor
    x = gait["fea3D"].transpose((3, 0, 1, 2))
    mpca = MPCA(var_ratio=var_ratio, vectorize=False)
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
    mpca.set_params(**{"vectorize": True, "n_components": n_components})

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
    x0_rec = mpca.inverse_transform(x0_proj.reshape(-1))
    testing.assert_equal(x0_rec.shape[1:], x[0].shape)

    # test n_components exceeds upper limit
    mpca.set_params(**{"vectorize": True, "n_components": np.prod(x.shape[1:]) + 1})
    x_proj = mpca.transform(x)
    testing.assert_equal(x_proj.shape[1], np.prod(mpca.shape_out))


def test_mpca_against_baseline(gait, baseline_model):
    x = gait["fea3D"].transpose((3, 0, 1, 2))
    baseline_proj_mats = [baseline_model["tUs"][i][0] for i in range(baseline_model["tUs"].size)]
    baseline_mean = baseline_model["TXmean"]
    mpca = MPCA(var_ratio=0.97)
    x_proj = mpca.fit(x).transform(x)
    testing.assert_allclose(baseline_mean, mpca.mean_)
    baseline_proj_x = multi_mode_dot(x - baseline_mean, baseline_proj_mats, modes=[1, 2, 3])
    # check whether the output embeddings is close to the baseline output by keeping the same variance ratio 97%
    testing.assert_allclose(x_proj**2, baseline_proj_x**2, rtol=relative_tol)
    # testing.assert_equal(x_proj.shape, baseline_proj_x.shape)

    for i in range(x.ndim - 1):
        # check whether each eigen-vector column is equal to/opposite of corresponding baseline eigen-vector column
        # testing.assert_allclose(abs(mpca.proj_mats[i]), abs(baseline_proj_mats[i]))
        testing.assert_allclose(mpca.proj_mats[i] ** 2, baseline_proj_mats[i] ** 2, rtol=relative_tol)


@pytest.mark.parametrize("num_components", [2, None])
def test_mida_shape_consistency(sample_data, num_components):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=num_components)
    mida.set_params(**mida.get_params())
    mida.fit(x, factors=factors)

    # Transform the whole data
    z = mida.transform(x, factors=factors)

    # If num_components is not None, check the shape of the transformed data
    if num_components is not None:
        testing.assert_equal(z.shape, (len(x), num_components))

    # Transform the source and target domain data separately
    (source_mask,) = np.where(domains != 0)
    z_src = mida.transform(x[source_mask], factors=factors[source_mask])
    z_tgt = mida.transform(x[~source_mask], factors=factors[~source_mask])
    # Check if transformations are consistent with separate domains
    testing.assert_allclose(z_src, z[source_mask])
    testing.assert_allclose(z_tgt, z[~source_mask])

    orig_coef_dim = mida.orig_coef_.shape[0]
    feature_dim = x.shape[1]
    assert mida.orig_coef_ is not None, "MIDA must have `orig_coef_` after fitting when kernel='linear'"
    assert orig_coef_dim == feature_dim, f"orig_coef_ shape mismatch: {orig_coef_dim} != {feature_dim}"


def test_mida_inverse_transform(sample_data):
    x, y, domains, factors = sample_data

    mida = MIDA(fit_inverse_transform=True)
    mida.fit(x, factors=factors)

    # Transform the whole data
    z = mida.transform(x, factors=factors)
    # Inverse transform the data
    x_rec = mida.inverse_transform(z)

    # We don't check whether the inverse transform is exactly equal to the original data
    # in terms of value since it is expected to be different due to the domain adaptation effect.
    # We only check the shape and dimensionality.
    assert len(x_rec) == len(x), f"Inverse transform failed: {len(x_rec)} != {len(x)}"
    assert x_rec.ndim == x.ndim, f"Inverse transform failed: {x_rec.ndim} != {x.ndim}"
    testing.assert_equal(x_rec.shape, x.shape)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
def test_mida_support_kernel(sample_data, kernel):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=8, kernel=kernel)
    mida.fit(x, factors=factors)

    is_linear = kernel == "linear"
    try:
        has_orig_coef = hasattr(mida, "orig_coef_")
        assert has_orig_coef, "MIDA must have `orig_coef_` after fitting when kernel='linear'"
    except NotImplementedError:
        assert not is_linear, "MIDA must not have `orig_coef_` after fitting when kernel!='linear'"

    # Transform the whole data
    z = mida.transform(x, factors=factors)

    # expect to allow multiple kernels supported
    assert mida._n_features_out == 8, f"Expected 8 components, got {mida.n_components_}"
    assert z.shape[1] == mida._n_features_out, f"Expected {z.shape[1]} features, got {mida._n_features_out}"


@pytest.mark.parametrize("augment", [True, False])
def test_mida_augment(sample_data, augment):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=8, kernel="linear", augment=augment, fit_inverse_transform=True)
    mida.fit(x, factors=factors)

    # expect validator for domain factors if augment=True
    has_validator = hasattr(mida, "_factor_validator")
    if augment:
        assert has_validator, "MIDA must have `_factor_validator` after fitting when augment=True"
        return

    assert not has_validator, "MIDA must not have `_factor_validator` after fitting when augment=False"


@pytest.mark.parametrize("ignore_y", [True, False])
def test_mida_ignore_y(sample_data, ignore_y):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=8, kernel="linear", ignore_y=ignore_y)
    mida.fit(x, y, factors=factors)

    # expect classes_ to be set if ignore_y=False
    has_classes = hasattr(mida, "classes_")
    if ignore_y:
        assert not has_classes, "MIDA must have `classes_` after fitting when ignore_y=True"
        return

    assert has_classes, "MIDA must not have `classes_` after fitting when ignore_y=False"


@pytest.mark.parametrize("eigen_solver", ["auto", "dense", "arpack", "randomized"])
def test_mida_eigen_solver(sample_data, eigen_solver):
    x, y, domains, factors = sample_data

    mida = MIDA(
        num_components=8,
        kernel="rbf",
        eigen_solver=eigen_solver,
        max_iter=200 if eigen_solver == "arpack" else None,
    )
    mida.fit(x, factors=factors)

    # Transform the whole data
    z = mida.transform(x, factors=factors)

    # expect the solver to have consistent number of components
    assert mida._n_features_out == 8, f"Expected 8 components, got {mida.n_components_}"
    assert z.shape[1] == mida._n_features_out, f"Expected {x.shape[1]} features, got {mida._n_features_out}"


@pytest.mark.parametrize("scale_components", [True, False])
def test_mida_scale_components(sample_data, scale_components):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=8, kernel="linear", scale_components=scale_components)
    mida.fit(x, factors=factors)

    # Transform the whole data
    z = mida.transform(x, factors=factors)

    # Expect the scale_components to have consistent number of components
    # the behavior expected is the zero eigenvalues component is masked, not indexed
    assert mida._n_features_out == 8, f"Expected 8 components, got {mida.n_components_}"
    assert z.shape[1] == mida._n_features_out, f"Expected {x.shape[1]} features, got {mida._n_features_out}"

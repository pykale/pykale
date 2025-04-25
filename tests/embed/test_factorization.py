import os

import numpy as np
import pytest
from numpy import testing
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("augment", [True, False])
def test_mida(num_components, kernel, augment):
    # Test an extreme case of domain shift
    # yet the data's manifold is linearly separable
    x, y, domains = make_domain_shifted_dataset(
        num_domains=10,
        num_samples_per_class=2,
        num_features=20,
        centroid_shift_scale=32768,
        random_state=0,
    )

    enc = OneHotEncoder(handle_unknown="ignore")
    factors = enc.fit_transform(domains.reshape(-1, 1)).toarray()
    mida = MIDA(num_components=num_components, kernel=kernel, augment=augment)
    mida_unsupervised = clone(mida)
    mida_semi_supervised = clone(mida)

    mida_unsupervised.set_params(**mida.get_params())
    mida_semi_supervised.set_params(**mida.get_params())

    (source_mask,) = np.where(domains != 0)

    # Mask the target domain labels
    y_masked = np.copy(y)
    y_masked[~source_mask] = -1

    mida_unsupervised.fit(x, factors=factors)
    mida_semi_supervised.fit(x, y_masked, factors=factors)

    # Transform the whole data
    z_unsupervised = mida_unsupervised.transform(x, factors=factors)
    z_semi_supervised = mida_semi_supervised.transform(x, factors=factors)

    # Transform the source and target domain data separately
    z_src_unsupervised = mida_unsupervised.transform(x[source_mask], factors=factors[source_mask])
    z_src_semi_supervised = mida_semi_supervised.transform(x[source_mask], factors=factors[source_mask])
    z_tgt_unsupervised = mida_unsupervised.transform(x[~source_mask], factors=factors[~source_mask])
    z_tgt_semi_supervised = mida_semi_supervised.transform(x[~source_mask], factors=factors[~source_mask])

    # Check if transformations are consistent with separate domains
    testing.assert_allclose(z_src_unsupervised, z_unsupervised[source_mask])
    testing.assert_allclose(z_src_semi_supervised, z_semi_supervised[source_mask])
    testing.assert_allclose(z_tgt_unsupervised, z_unsupervised[~source_mask])
    testing.assert_allclose(z_tgt_semi_supervised, z_semi_supervised[~source_mask])

    if num_components is not None:
        # We only test when num_components is not None since it can vary
        # given the eigenvalues.
        testing.assert_equal(z_unsupervised.shape, (len(x), num_components))
        testing.assert_equal(z_semi_supervised.shape, (len(x), num_components))
    else:
        # We only test for prediction performance when we can leverage all the components
        # Expect the domain shifted dataset to perform better when mida applied

        # Logistic regression is used since it is a linearly separable classifier
        # and the dataset is linearly separable

        # Train on the source domain, score on the target domain
        classifier = LogisticRegression(random_state=0, max_iter=10)
        classifier.fit(x[source_mask], y[source_mask])
        score_tgt = classifier.score(x[~source_mask], y[~source_mask])

        # Train on the source domain, score on the target domain
        # using MIDA-transformed data
        classifier.fit(z_src_unsupervised, y[source_mask])
        score_tgt_unsupervised = classifier.score(z_tgt_unsupervised, y[~source_mask])

        # Train on the source domain, score on the target domain
        # using SMIDA-transformed data
        classifier.fit(z_src_semi_supervised, y[source_mask])
        score_tgt_semi_supervised = classifier.score(z_tgt_semi_supervised, y[~source_mask])

        assert (
            score_tgt_unsupervised > score_tgt
        ), f"MIDA did not improve target domain performance MIDA ({score_tgt_unsupervised:.4f}) > Baseline ({score_tgt:.4f})"
        assert (
            score_tgt_semi_supervised > score_tgt
        ), f"SMIDA did not improve target domain performance SMIDA ({score_tgt_semi_supervised:.4f}) > Baseline ({score_tgt:.4f})"

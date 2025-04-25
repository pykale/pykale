# import numpy as np
import pytest

# from numpy import testing
# from sklearn.datasets import make_blobs
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from kale.pipeline.mida_trainer import MIDATrainer

from ..helpers.toy_dataset import make_domain_shifted_dataset


@pytest.fixture(scope="module")
def toy_data():
    # Test an extreme case of domain shift
    # yet the data's manifold is linearly separable
    x, y, domains = make_domain_shifted_dataset(
        num_domains=10,
        num_samples_per_class=2,
        num_features=20,
        centroid_shift_scale=32768,
        random_state=0,
    )

    return x, y, domains


TRANSFORMER = [None, MinMaxScaler()]
PARAM_GRID = [
    {"C": [2**-5, 1, 2**15]},
    {"C": [2**-5, 1, 2**15], "domain_adapter__mu": [2**-5, 1, 2**15]},
]


@pytest.mark.parametrize("transformer", TRANSFORMER)
@pytest.mark.parametrize("param_grid", PARAM_GRID)
@pytest.mark.parametrize("search_strategy", ["random", "grid"])
@pytest.mark.parametrize("scoring", [None, "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]])
@pytest.mark.parametrize("cv", [None, 3, LeaveOneGroupOut()])
def test_mida_trainer(toy_data, transformer, param_grid, search_strategy, scoring, cv):
    x, y, domains = toy_data

    param_grid = clone(param_grid, safe=False)
    if transformer is not None:
        param_grid.update({"transformer__feature_range": [(-1, 1), (0, 1)]})

    refit = True
    if isinstance(scoring, list):
        refit = "roc_auc"

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        transformer=transformer,
        param_grid=param_grid,
        search_strategy=search_strategy,
        scoring=scoring,
        cv=cv,
        num_iter=10,
        refit=refit,
    )

    factors = OneHotEncoder(sparse_output=False).fit_transform(domains.reshape(-1, 1))
    trainer.fit(x, y, factors=factors, groups=domains)

    assert "params" in trainer.cv_results_
    assert all(k in param_grid for k in trainer.best_params_.keys())
    if isinstance(scoring, list):
        for score in scoring:
            assert f"mean_test_{score}" in trainer.cv_results_
            assert f"std_test_{score}" in trainer.cv_results_
            assert f"rank_test_{score}" in trainer.cv_results_
    else:
        assert "mean_test_score" in trainer.cv_results_
        assert "std_test_score" in trainer.cv_results_
        assert "rank_test_score" in trainer.cv_results_

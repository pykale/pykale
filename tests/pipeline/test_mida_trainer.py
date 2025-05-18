# import numpy as np
import pytest
from numpy import testing
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from kale.pipeline.mida_trainer import MIDATrainer

from ..helpers.toy_dataset import make_domain_shifted_dataset

TRANSFORMER = [None, MinMaxScaler()]
PARAM_GRID = [
    {"C": [2**-5, 1, 2**15]},
    {"C": [2**-5, 1, 2**15], "domain_adapter__mu": [2**-5, 1, 2**15]},
    [
        {"C": [2**-5, 1, 2**15]},
        {"C": [2**-5, 1, 2**15], "domain_adapter__mu": [2**-5, 1, 2**15]},
    ],
]
SEARCH_STRATEGY = ["random", "grid"]
SCORING = [None, "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]]
CV = [None, 3, LeaveOneGroupOut()]


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

    factors = OneHotEncoder(sparse_output=False).fit_transform(domains.reshape(-1, 1))

    return x, y, domains, factors


@pytest.mark.parametrize("transformer", TRANSFORMER)
@pytest.mark.parametrize("param_grid", PARAM_GRID)
def test_mida_trainer_params_consistency(toy_data, transformer, param_grid):
    x, y, domains, factors = toy_data

    param_grid = clone(param_grid, safe=False)
    if transformer is not None:
        if isinstance(param_grid, list):
            for i in range(len(param_grid)):
                param_grid[i].update({"transformer__feature_range": [(-1, 1), (0, 1)]})
        else:
            param_grid.update({"transformer__feature_range": [(-1, 1), (0, 1)]})

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        transformer=transformer,
        param_grid=param_grid,
        error_score="raise",
        random_state=0,
    )

    trainer.fit(x, y, factors=factors, groups=domains)

    # gather keys from list of param_grid
    if isinstance(param_grid, list):
        param_keys = set(k for d in param_grid for k in d.keys())
    else:
        param_keys = set(param_grid.keys())

    missing_param_keys = set(trainer.best_params_.keys()) - param_keys

    assert "params" in trainer.cv_results_, "Missing 'params' in cv_results_"
    assert len(missing_param_keys) == 0, f"Missing {missing_param_keys} in best_params_"


@pytest.mark.parametrize("scoring", [None, "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]])
def test_mida_trainer_scoring_support(toy_data, scoring):
    x, y, domains, factors = toy_data

    refit = True
    if isinstance(scoring, list):
        refit = "roc_auc"

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        param_grid=PARAM_GRID[0],
        scoring=scoring,
        refit=refit,
        error_score="raise",
    )

    trainer.fit(x, y, factors=factors, groups=domains)

    if scoring is None or isinstance(scoring, str):
        scoring = ["score"]

    for score in scoring:
        assert f"mean_test_{score}" in trainer.cv_results_, f"Missing mean_test_{score} in cv_results_"
        assert f"std_test_{score}" in trainer.cv_results_, f"Missing std_test_{score} in cv_results_"
        assert f"rank_test_{score}" in trainer.cv_results_, f"Missing rank_test_{score} in cv_results_"


@pytest.mark.parametrize("cv", [None, 3, LeaveOneGroupOut()])
def test_mida_trainer_cv_support(toy_data, cv):
    x, y, domains, factors = toy_data

    trainer = MIDATrainer(
        LogisticRegression(random_state=0, max_iter=10), param_grid=PARAM_GRID[0], cv=cv, error_score="raise"
    )

    trainer.fit(x, y, factors=factors, groups=domains)

    if cv is None:
        cv = 5
    elif isinstance(cv, LeaveOneGroupOut):
        cv = 10

    assert any(
        key.startswith(f"split{cv-1}") for key in trainer.cv_results_.keys()
    ), f"Missing keys starting with 'split{cv-1}' in cv_results_"


@pytest.mark.parametrize("search_strategy", ["random", "grid"])
def test_mida_trainer_search_strategy_support(toy_data, search_strategy):
    x, y, domains, factors = toy_data

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        param_grid=PARAM_GRID[0],
        search_strategy=search_strategy,
        error_score="raise",
    )
    trainer.fit(x, y, factors=factors, groups=domains)

    # assert len(trainer.)


def test_mida_trainer_fit_and_methods(toy_data):
    x, y, domains, factors = toy_data

    param_grid = clone(PARAM_GRID[1], safe=False)
    param_grid.update(
        {
            "transformer__feature_range": [(-1, 1), (0, 1)],
            "domain_adapter__num_components": [10],
            "domain_adapter__fit_inverse_transform": [True],
        }
    )

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        transformer=MinMaxScaler(),
        param_grid=param_grid,
        error_score="raise",
    )

    trainer.fit(x, y, factors=factors, groups=domains)

    # check n_features_in_
    assert hasattr(trainer, "n_features_in_"), "n_features_in_ should be set"
    assert trainer.n_features_in_ == x.shape[1], f"n_features_in_ should be {x.shape[1]}, got {trainer.n_features_in_}"

    # test adaptation (excluding estimator)
    x_transformed = trainer.adapt(x, factors=factors)
    testing.assert_array_equal((len(x), 10), x_transformed.shape)

    # test predict
    y_pred = trainer.predict(x)
    testing.assert_array_equal((len(x),), y_pred.shape)

    # test predict_proba
    y_proba = trainer.predict_proba(x)
    testing.assert_array_equal((len(x), 2), y_proba.shape)

    # test predict_log_proba
    y_log_proba = trainer.predict_log_proba(x)
    testing.assert_array_equal((len(x), 2), y_log_proba.shape)

    # test decision_function
    y_decision = trainer.decision_function(x)
    testing.assert_array_equal((len(x),), y_decision.shape)

    # test score
    score = trainer.score(x, y)
    assert isinstance(score, float), f"Score should be a float, got '{type(score)}' instead"

    # test unsupervised models
    trainer = trainer.set_params(
        estimator=PCA(n_components=2, random_state=0), param_grid={"domain_adapter__fit_inverse_transform": [True]}
    )
    trainer.fit(x, factors=factors, groups=domains)

    # test score_sample
    score = trainer.score_samples(x)
    testing.assert_array_equal((len(x),), score.shape)

    # test transform
    transform = trainer.transform(x, factors=factors)
    testing.assert_array_equal((len(x), 2), transform.shape)

    # test inverse_transform
    inv_transform = trainer.inverse_transform(transform)
    testing.assert_array_equal((len(x), 20), inv_transform.shape)

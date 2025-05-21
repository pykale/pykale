import numpy as np
import pytest
from numpy import testing

# from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from kale.pipeline.multi_domain_adapter import (
    AutoMIDAClassificationTrainer,
    CLASSIFIER_PARAMS,
    MIDA_PARAMS,
    MIDATrainer,
)

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
NUM_DOMAINS = 4


@pytest.fixture(scope="module")
def toy_data():
    # Test an extreme case of domain shift
    # yet the data's manifold is linearly separable
    x, y, domains = make_domain_shifted_dataset(
        num_domains=NUM_DOMAINS,
        num_samples_per_class=4,
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

    param_grid = param_grid.copy()
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

    trainer.fit(x, y, group_labels=domains)

    # gather keys from a list of param_grid
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

    trainer.fit(x, y, group_labels=domains)

    if scoring is None or isinstance(scoring, str):
        scoring = ["score"]

    for score in scoring:
        assert f"mean_test_{score}" in trainer.cv_results_, f"Missing mean_test_{score} in cv_results_"
        assert f"std_test_{score}" in trainer.cv_results_, f"Missing std_test_{score} in cv_results_"
        assert f"rank_test_{score}" in trainer.cv_results_, f"Missing rank_test_{score} in cv_results_"


@pytest.mark.parametrize("use_mida", [True, False])
def test_mida_trainer_use_mida(toy_data, use_mida):
    x, y, domains, factors = toy_data

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        param_grid=PARAM_GRID[0],
        use_mida=use_mida,
        error_score="raise",
    )

    trainer.fit(x, y, group_labels=factors, groups=domains)

    if use_mida:
        assert hasattr(trainer, "best_mida_"), "MIDA should be set"
    else:
        assert not hasattr(trainer, "best_mida_"), "MIDA should not be set"


@pytest.mark.parametrize("cv", [None, 3, LeaveOneGroupOut()])
def test_mida_trainer_cv_support(toy_data, cv):
    x, y, domains, factors = toy_data

    trainer = MIDATrainer(
        LogisticRegression(random_state=0, max_iter=10), param_grid=PARAM_GRID[0], cv=cv, error_score="raise"
    )

    trainer.fit(x, y, group_labels=domains)

    if cv is None:
        cv = 5
    elif isinstance(cv, LeaveOneGroupOut):
        cv = NUM_DOMAINS

    assert any(
        key.startswith(f"split{cv - 1}") for key in trainer.cv_results_.keys()
    ), f"Missing keys starting with 'split{cv - 1}' in cv_results_"


@pytest.mark.parametrize("search_strategy", ["random", "grid"])
def test_mida_trainer_search_strategy_support(toy_data, search_strategy):
    x, y, domains, factors = toy_data

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        param_grid=PARAM_GRID[0],
        search_strategy=search_strategy,
        error_score="raise",
    )
    trainer.fit(x, y, group_labels=factors, groups=domains)

    # assert len(trainer.)


def test_mida_trainer_fit_and_methods(toy_data):
    x, y, domains, factors = toy_data

    param_grid = PARAM_GRID.copy()[1]
    param_grid.update(
        {
            "transformer__feature_range": [(-1, 1), (0, 1)],
            "domain_adapter__num_components": [4],
            "domain_adapter__fit_inverse_transform": [True],
        }
    )

    trainer = MIDATrainer(
        estimator=LogisticRegression(random_state=0, max_iter=10),
        transformer=MinMaxScaler(),
        param_grid=param_grid,
        error_score="raise",
    )

    trainer.fit(x, y, group_labels=domains)

    # check n_features_in_
    assert hasattr(trainer, "n_features_in_"), "n_features_in_ should be set"
    assert trainer.n_features_in_ == x.shape[1], f"n_features_in_ should be {x.shape[1]}, got {trainer.n_features_in_}"

    # test adaptation (excluding estimator)
    x_transformed = trainer.adapt(x, group_labels=domains)
    testing.assert_array_equal((len(x), 4), x_transformed.shape)

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
        **{
            "estimator": PCA(n_components=2, random_state=0),
            "param_grid": {"domain_adapter__fit_inverse_transform": [True]},
        }
    )
    trainer.fit(x, group_labels=domains)

    # test score_sample
    score = trainer.score_samples(x)
    testing.assert_array_equal((len(x),), score.shape)

    # test transform
    transform = trainer.transform(x)
    testing.assert_array_equal((len(x), 2), transform.shape)

    # test inverse_transform
    inv_transform = trainer.inverse_transform(transform)
    testing.assert_array_equal((len(x), 20), inv_transform.shape)


@pytest.mark.parametrize("search_strategy", ["grid", "random"])
def test_auto_mida_trainer_basic(toy_data, search_strategy, monkeypatch):
    x, y, domains, factors = toy_data

    if search_strategy == "grid":
        # Monkeypatch classifier params to limit the grid
        monkeypatch.setitem(CLASSIFIER_PARAMS["lr"], "C", [1])
        monkeypatch.setitem(MIDA_PARAMS, "domain_adapter__mu", [1])

    trainer = AutoMIDAClassificationTrainer(
        classifier="lr",
        search_strategy=search_strategy,
        scoring="accuracy",
        num_search_iter=2,
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=domains)

    assert hasattr(trainer, "best_classifier_")
    assert hasattr(trainer, "best_score_")
    assert isinstance(trainer.predict(x, group_labels=domains), np.ndarray)

    # Check delegation
    assert trainer.score(x, y, domains) > 0
    assert trainer.predict_proba(x, domains).shape == (len(x), 2)
    assert trainer.adapt(x, domains).shape[1] <= (x.shape[1] + factors.shape[1])


@pytest.mark.parametrize("classifier", ["linear_svm", "svm", "ridge"])
def test_auto_mida_trainer_classifier_selection(toy_data, classifier, monkeypatch):
    x, y, domains, factors = toy_data

    # Monkeypatch classifier param grid to limit to a single value for speed
    classifier_param = list(CLASSIFIER_PARAMS[classifier].keys())[0]
    monkeypatch.setitem(CLASSIFIER_PARAMS[classifier], classifier_param, [1])

    trainer = AutoMIDAClassificationTrainer(
        classifier=classifier,
        search_strategy="random",
        scoring="accuracy",
        num_search_iter=1,
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=domains)
    assert trainer.best_classifier_ is not None
    assert trainer.best_params_ is not None


@pytest.mark.parametrize("transformer", [MinMaxScaler(), None])
@pytest.mark.parametrize("use_mida", [True, False])
def test_auto_mida_trainer_property_accessors(toy_data, transformer, use_mida, monkeypatch):
    x, y, domains, factors = toy_data

    monkeypatch.setitem(CLASSIFIER_PARAMS["lr"], "C", [1])
    monkeypatch.setitem(MIDA_PARAMS, "domain_adapter__mu", [1])

    trainer = AutoMIDAClassificationTrainer(
        classifier="lr",
        transformer=transformer,
        use_mida=use_mida,
        search_strategy="random",
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=domains)

    # Always expected
    _ = trainer.best_classifier_
    _ = trainer.best_score_
    _ = trainer.best_params_
    _ = trainer.best_index_
    _ = trainer.cv_results_
    _ = trainer.scorer_
    _ = trainer.n_splits_
    _ = trainer.refit_time_
    _ = trainer.multimetric_
    _ = trainer.n_features_in_

    if hasattr(trainer.trainer_, "feature_names_in_"):
        _ = trainer.trainer_.feature_names_in_

    if transformer is not None:
        _ = trainer.best_transformer_

    if use_mida:
        _ = trainer.best_mida_


# @pytest.mark.parametrize("use_mida", [True, False])
# @pytest.mark.parametrize("classifier", CLASSIFIER_PARAMS.keys())
# @pytest.mark.parametrize("augment", [None])
@pytest.mark.parametrize("augment", ["pre", "post", None])
# def test_auto_mida_trainer_coef_shape(toy_data, use_mida, classifier, augment, monkeypatch):
def test_auto_mida_trainer_coef_shape(toy_data, augment, monkeypatch):
    x, y, domains, factors = toy_data

    monkeypatch.setitem(CLASSIFIER_PARAMS["lr"], "C", [1])
    monkeypatch.setitem(MIDA_PARAMS, "mu", [1])
    monkeypatch.setitem(MIDA_PARAMS, "augment", [augment])
    monkeypatch.setitem(MIDA_PARAMS, "ignore_y", [True])

    trainer = AutoMIDAClassificationTrainer(
        classifier="lr",
        use_mida=True,
        search_strategy="random",
        scoring="accuracy",
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=domains)

    coef = trainer.coef_

    feature_dim = x.shape[1]
    if augment is not None:
        feature_dim += factors.shape[1]

    assert coef.shape == (1, feature_dim), f"Expected shape (1, {feature_dim}), got {coef.shape}"


# Test nonlinear=True with a fixed classifier ("svm")
def test_auto_mida_trainer_nonlinear_enabled(toy_data, monkeypatch):
    x, y, domains, factors = toy_data

    monkeypatch.setitem(CLASSIFIER_PARAMS["svm"], "C", [1])
    monkeypatch.setitem(MIDA_PARAMS, "domain_adapter__mu", [1])
    monkeypatch.setitem(MIDA_PARAMS, "domain_adapter__kernel", ["linear", "rbf"])

    trainer = AutoMIDAClassificationTrainer(
        classifier="svm",
        nonlinear=True,
        search_strategy="random",
        scoring="accuracy",
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=factors)

    assert trainer.best_classifier_ is not None
    assert trainer.best_params_ is not None
    assert trainer.best_score_ is not None
    with pytest.raises(ValueError, match="coef_ is not available when `nonlinear=True`."):
        _ = trainer.coef_


# Test classifier="auto" without nonlinear
def test_auto_mida_trainer_classifier_auto(toy_data, monkeypatch):
    x, y, domains, factors = toy_data

    for name in CLASSIFIER_PARAMS:
        param_name = list(CLASSIFIER_PARAMS[name].keys())[0]
        monkeypatch.setitem(CLASSIFIER_PARAMS[name], param_name, [1])

    monkeypatch.setitem(MIDA_PARAMS, "domain_adapter__mu", [1])

    trainer = AutoMIDAClassificationTrainer(
        classifier="auto",
        search_strategy="random",
        scoring="accuracy",
        num_solver_iter=10,
        cv=2,
        error_score="raise",
        random_state=0,
    )
    trainer.fit(x, y, group_labels=factors)

    assert trainer.best_classifier_ is not None
    assert trainer.best_params_ is not None
    assert trainer.best_score_ is not None

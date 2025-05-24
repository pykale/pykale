import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kale.embed.factorization import MIDA
from kale.evaluate import cross_validation
from kale.pipeline.multi_domain_adapter import CoIRLS

from ..helpers.toy_dataset import make_domain_shifted_dataset


@pytest.fixture(scope="module")
def sample_data():
    # Sample data for testing
    x, y, groups = make_domain_shifted_dataset(
        num_domains=2,
        num_samples_per_class=100,
        num_features=40,
        class_sep=1.0,
        centroid_shift_scale=5.0,
        random_state=0,
    )

    # Convert labels to strings for testing
    groups = np.where(groups == 0, "A", "B")
    factors = OneHotEncoder(sparse_output=False).fit_transform(groups.reshape(-1, 1))

    return x, y, groups, factors


# Checks leave-one-group-out results for above sample data
def check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False):
    x, y, groups, _ = sample_data
    results = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation)

    # Check if all keys are present in the result dictionary
    assert "Target" in results
    assert "Num_samples" in results
    assert "Accuracy" in results

    # Check if the length of 'Target', 'Num_samples', and 'Accuracy' lists is the same
    assert len(results["Target"]) == len(results["Num_samples"]) == len(results["Accuracy"])

    # Check if the last element of 'Target' is "Average"
    assert results["Target"] == ["A", "B", "Average"]

    # Check if the number of samples is correct for each target group
    assert results["Num_samples"] == [200, 200, 400]

    # Check if the accuracy scores are within the expected range
    for accuracy in results["Accuracy"]:
        assert 0 <= accuracy <= 1


def test_leave_one_group_out_without_domain_adaptation(sample_data):
    estimator = DummyClassifier()
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False)


def test_leave_one_group_out_with_domain_adaptation(sample_data):
    estimator = CoIRLS(kernel="linear", lambda_=1.0, alpha=1.0)
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=True)


@pytest.mark.parametrize("cv", [None, 3, LeaveOneGroupOut()])
def test_cross_validate_cv_supports(sample_data, cv):
    x, y, groups, factors = sample_data
    estimator = DummyClassifier()

    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=None,
        domain_adapter=None,
        group_labels=factors,
        scoring=None,
        cv=cv,
        fit_args=None,
        return_train_score=False,
        return_indices=False,
        error_score="raise",
    )

    if cv is None:
        cv = 5

    if isinstance(cv, LeaveOneGroupOut):
        cv = 2

    assert len(results["test_score"]) == cv, f'Expected {cv} folds, got {len(results["test_score"])}'


@pytest.mark.parametrize("transformer", [None, StandardScaler()])
@pytest.mark.parametrize("domain_adapter", [None, MIDA()])
def test_cross_validate_with_transformer_and_domain_adaptation(sample_data, transformer, domain_adapter):
    x, y, groups, factors = sample_data

    estimator = DummyClassifier()
    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=transformer,
        domain_adapter=domain_adapter,
        group_labels=factors,
        scoring=None,
        fit_args=None,
        return_train_score=False,
        return_estimator=False,
        return_indices=False,
        error_score="raise",
    )

    assert "fit_time" in results, "fit_time not in results"
    assert "score_time" in results, "score_time not in results"
    assert "test_score" in results, "test_score not in results"


@pytest.mark.parametrize("scoring", [None, "f1", "roc_auc", ["accuracy", "f1", "roc_auc"]])
def test_cross_validate_scoring_support(sample_data, scoring):
    x, y, groups, factors = sample_data

    estimator = DummyClassifier()
    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=None,
        domain_adapter=None,
        scoring=scoring,
        fit_args=None,
        return_train_score=False,
        return_estimator=False,
        return_indices=False,
        error_score="raise",
    )

    if scoring is None or isinstance(scoring, str):
        scoring = ["score"]

    for score in scoring:
        assert f"test_{score}" in results, f"Missing test_{score} in cv_results_"


@pytest.mark.parametrize("return_indices", [True, False])
def test_cross_validate_return_indices(sample_data, return_indices):
    x, y, groups, factors = sample_data

    estimator = DummyClassifier()
    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=None,
        domain_adapter=None,
        scoring=None,
        fit_args=None,
        return_train_score=False,
        return_estimator=False,
        return_indices=return_indices,
        error_score="raise",
    )

    if return_indices:
        assert "indices" in results, "Expected 'indices' in results"
        return

    assert "indices" not in results, "Did not expect 'indices' in results"


@pytest.mark.parametrize("return_train_score", [True, False])
def test_cross_validate_return_train_score(sample_data, return_train_score):
    x, y, groups, factors = sample_data

    estimator = DummyClassifier()
    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=None,
        domain_adapter=None,
        group_labels=factors,
        scoring=None,
        fit_args=None,
        return_train_score=return_train_score,
        return_estimator=False,
        return_indices=False,
        error_score="raise",
    )

    if return_train_score:
        assert "train_score" in results, "Expected 'train_score' in results"
        return

    assert "train_score" not in results, "Did not expect 'train_score' in results"


@pytest.mark.parametrize("return_estimator", [True, False])
def test_cross_validate_return_estimator(sample_data, return_estimator):
    x, y, groups, factors = sample_data

    estimator = DummyClassifier()
    results = cross_validation.cross_validate(
        estimator,
        x,
        y,
        groups=groups,
        transformer=None,
        domain_adapter=None,
        group_labels=factors,
        scoring=None,
        fit_args=None,
        return_train_score=False,
        return_estimator=return_estimator,
        return_indices=False,
        error_score="raise",
    )

    if return_estimator:
        assert "estimator" in results, "Expected 'estimator' in results"
        return

    assert "estimator" not in results, "Did not expect 'estimator' in results"


@pytest.mark.parametrize("error_score", [np.nan, "raise", -1])
def test_cross_validate_error_score(sample_data, error_score):
    x, y, groups, factors = sample_data
    # Ensure y is float to cause an error
    y = np.random.normal(size=y.shape).astype(float)

    estimator = DummyClassifier()

    try:
        results = cross_validation.cross_validate(
            estimator,
            x,
            y,
            groups=groups,
            transformer=None,
            domain_adapter=None,
            group_labels=factors,
            scoring=None,
            fit_args=None,
            return_train_score=False,
            return_estimator=False,
            return_indices=False,
            error_score=error_score,
        )

        if np.isnan(error_score):
            assert np.all(np.isnan(results["test_score"])), "Expected all NaN in test_score"
        else:
            assert np.all(results["test_score"] == error_score), "Expected all -1 in test_score"

    except ValueError as e:
        assert len(e.args) == 1, "Expected raise when error_score='raise'"

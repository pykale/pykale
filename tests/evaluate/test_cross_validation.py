import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from kale.evaluate import cross_validation
from kale.pipeline.multi_domain_adapter import CoIRLS


@pytest.fixture
def sample_data():
    # Sample data for testing
    x = np.array([np.random.rand(100)] * 8)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
    return x, y, groups


# Checks leave-one-group-out results for above sample data
def check_leave_one_group_out_results(results):
    assert "Target" in results
    assert "Num_samples" in results
    assert "Accuracy" in results
    assert len(results["Target"]) == len(results["Num_samples"]) == len(results["Accuracy"])
    assert results["Target"] == ["A", "B", "Average"]
    assert results["Num_samples"] == [4, 4, 8]
    for accuracy in results["Accuracy"]:
        assert 0 <= accuracy <= 1


def test_leave_one_group_out_without_domain_adaptation(sample_data):
    x, y, groups = sample_data
    estimator = DummyClassifier()
    results = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation=False)
    check_leave_one_group_out_results(results)


def test_leave_one_group_out_with_domain_adaptation(sample_data):
    x, y, groups = sample_data
    estimator = CoIRLS(kernel="linear", lambda_=1.0, alpha=1.0)
    results = cross_validation.leave_one_group_out(x, y, groups, estimator, domain_adaptation=True)
    check_leave_one_group_out_results(results)

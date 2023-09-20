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
def check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False):
    x, y, groups = sample_data
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
    assert results["Num_samples"] == [4, 4, 8]

    # Check if the accuracy scores are within the expected range
    for accuracy in results["Accuracy"]:
        assert 0 <= accuracy <= 1


def test_leave_one_group_out_without_domain_adaptation(sample_data):
    estimator = DummyClassifier()
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=False)


def test_leave_one_group_out_with_domain_adaptation(sample_data):
    estimator = CoIRLS(kernel="linear", lambda_=1.0, alpha=1.0)
    check_leave_one_group_out_results(sample_data, estimator, domain_adaptation=True)

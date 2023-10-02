"""
Functions implementing cross-validation methods for assessing model fit.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder


def leave_one_group_out(x, y, groups, estimator, use_domain_adaptation=False) -> dict:
    """
    Perform leave one group out cross validation for a given estimator.

    Args:
        x (np.ndarray or torch.tensor): Input data [n_samples, n_features].
        y (np.ndarray or torch.tensor): Target labels [n_samples].
        groups (np.ndarray or torch.tensor): Group labels to be left out [n_samples].
        estimator (estimator object): Machine learning estimator to be evaluated from kale or scikit-learn.
        use_domain_adaptation (bool): Whether to use domain adaptation, i.e., leveraging test data, during training.

    Returns:
        dict: A dictionary containing results for each target group with 3 keys.
            - 'Target': A list of unique target groups or classes. The final entry is "Average".
            - 'Num_samples': A list where each entry indicates the number of samples in its corresponding target group.
                            The final entry represents the total number of samples.
            - 'Accuracy': A list where each entry indicates the accuracy score for its corresponding target group.
                            The final entry represents the overall mean accuracy.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    group_mat = enc.fit_transform(groups.reshape(-1, 1)).toarray()
    target, num_samples, accuracy = np.unique(groups).tolist(), [], []
    y_pred = np.zeros(y.shape)  # Store all predicted labels to compute accuracy

    for train, test in LeaveOneGroupOut().split(x, y, groups=groups):
        x_src, x_tgt = x[train], x[test]
        y_src, y_tgt = y[train], y[test]

        if use_domain_adaptation:
            estimator.fit(np.concatenate((x_src, x_tgt)), y_src, np.concatenate((group_mat[train], group_mat[test])))
        else:
            estimator.fit(x_src, y_src)

        y_pred[test] = estimator.predict(x_tgt)
        num_samples.append(x_tgt.shape[0])
        accuracy.append(accuracy_score(y_tgt, y_pred[test]))

    target.append("Average")
    num_samples.append(x.shape[0])
    accuracy.append(accuracy_score(y, y_pred))

    return {
        "Target": target,
        "Num_samples": num_samples,
        "Accuracy": accuracy,
    }

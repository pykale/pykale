import numpy as np
from sklearn.utils.validation import check_random_state


def make_domain_shifted_dataset(
    n_domains=3,
    n_samples_per_class=100,
    n_features=2,
    class_sep=1.0,
    centroid_shift_scale=5.0,
    random_state=None,
):
    random_state = check_random_state(random_state)

    # Shared linear separator
    w = random_state.randn(n_features)
    w = w / np.linalg.norm(w)

    X_all = []
    y_all = []
    domain_all = []

    for i_domain in range(n_domains):
        domain_shift = random_state.randn(n_features) * centroid_shift_scale

        for label in [0, 1]:
            class_mean = (label - 0.5) * class_sep * w + domain_shift
            cov = np.eye(n_features)
            X_class = random_state.multivariate_normal(class_mean, cov, n_samples_per_class)
            y_class = np.full(n_samples_per_class, label)
            domain_class = np.full(n_samples_per_class, i_domain)

            X_all.append(X_class)
            y_all.append(y_class)
            domain_all.append(domain_class)

    x = np.vstack(X_all)
    y = np.concatenate(y_all)
    domains = np.concatenate(domain_all)

    idx = random_state.permutation(len(x))
    x = x[idx]
    y = y[idx]
    domains = domains[idx]

    return x, y, domains

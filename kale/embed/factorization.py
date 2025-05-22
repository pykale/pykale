# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Python implementation of a tensor factorization algorithm Multilinear Principal Component Analysis (MPCA)
and a matrix factorization algorithm Maximum Independence Domain Adaptation (MIDA）
"""
import logging
import warnings
from numbers import Integral, Real

import numpy as np
import scipy.linalg as la
from numpy.linalg import multi_dot
from scipy.sparse.linalg import eigsh
from sklearn.base import _fit_context, BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.preprocessing import FunctionTransformer, KernelCenterer, OneHotEncoder
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_eigsh, safe_sparse_dot, svd_flip
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_psd_eigenvalues,
    _num_features,
    _num_samples,
    check_is_fitted,
    NotFittedError,
    validate_data,
)
from tensorly.base import fold, unfold
from tensorly.tenalg import multi_mode_dot


def _check_n_dim(x, n_dims):
    """Raise error if the dimension of the input data is inconsistent with the expected value.

    Args:
        x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)
        n_dims (int): number of dimensions expected, i.e. N+1

    """
    if not x.ndim == n_dims:
        error_msg = "The expected number of dimensions is %s but it is %s for given data" % (n_dims, x.ndim)
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_shape(x, shape_):
    """Raise error if the shape for each sample (i.e. excluding the first dimension) of the input data is not consistent
        with the given shape.

    Args:
        x (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)
        shape_: expected shape for each sample, i.e. (I_1, I_2, ..., I_N)

    """
    if not x.shape[1:] == shape_:
        error_msg = "The expected shape of data is %s, but %s for given data" % (x.shape[1:], shape_)
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_tensor_dim_shape(x, n_dims, shape_):
    """Check whether the number of dimensions of the input data and the shape for each sample are consistent with
        expected values

    Args:
        x (array-like): input data, shape (n_samples, I_1, I_2, ..., I_N)
        n_dims (int): number of dimensions expected, i.e. N+1
        shape_: expected shape for each sample, i.e. (I_1, I_2, ..., I_N)

    """
    _check_n_dim(x, n_dims)
    _check_shape(x, shape_)


def _centering_kernel(size, dtype=np.float64):
    """Generate a centering kernel matrix.

    Args:
        size (int): The size of the kernel matrix.
        dtype (data-type, optional): The desired data-type for the kernel matrix. Default is np.float64.
    Returns:
        array-like: The centering kernel matrix of shape (size, size).
    """

    identity = np.eye(size, dtype=dtype)
    ones_matrix = np.ones((size, size), dtype)
    return identity - ones_matrix / size


def _check_num_components(k, num_components):
    """Check and set the number of components to keep after eigendecomposition.

    Args:
        k (array-like): The kernel matrix.
        num_components (int, optional): The number of components to keep. If None, all components are kept.
    Returns:
        int: The number of components to keep.
    """
    k_size = _num_features(k)
    if num_components is None:
        num_components = k_size  # use all dimensions
    else:
        num_components = min(k_size, num_components)
    return num_components


# solver helper


def _check_solver(k, num_components, solver):
    """Check and set the solver for eigendecomposition.
    Args:
        k (array-like): The kernel matrix.
        num_components (int): The number of components to keep.
        solver (str): The solver to use. Can be 'auto', 'arpack', 'randomized', or 'dense'.
    Returns:
        str: The solver to use.
    """
    k_size = _num_features(k)

    if solver == "auto" and k_size > 200 and num_components < 10:
        solver = "arpack"
    elif solver == "auto":
        solver = "dense"
    return solver


def _eigendecompose(
    k,
    num_components=None,
    solver="auto",
    random_state=None,
    max_iter=None,
    tol=0,
    iterated_power="auto",
):
    """Eigendecompose the kernel matrix.
    Args:
        k (array-like): The kernel matrix.
        num_components (int, optional): The number of components to keep. If None, all components are kept.
        solver (str): The solver to use. Can be 'auto', 'arpack', 'randomized', or 'dense'.
        random_state (int or np.random.RandomState, optional): Random seed for reproducibility.
        max_iter (int, optional): Maximum number of iterations for the solver.
        tol (float, optional): Tolerance for convergence.
        iterated_power (int or str, optional): Number of iterations for randomized solver.
    Returns:
        tuple: `(eigenvalues, eigenvectors)` of the kernel matrix.
    """
    # we accept tuple or list for k, in case a method
    # need to use a generalized eigenvalue decomposition
    if isinstance(k, (tuple, list)):
        a, b = k
    else:
        a, b = k, None

    num_components = _check_num_components(k, num_components)
    solver = _check_solver(k, num_components, solver)

    if solver == "arpack":
        v0 = _init_arpack_v0(_num_features(a), random_state)
        return eigsh(
            a,
            num_components,
            b,
            which="LA",
            tol=tol,
            maxiter=max_iter,
            v0=v0,
        )

    if solver == "randomized":
        # To support methods that require generalized eigendecomposition,
        # for randomized solver that doesn't support it by default.
        # We use the inverse of b to transform a to obtain an equivalent
        # formulation using regular eigendecomposition.
        if b is not None:
            a = la.inv(b) @ a

        return _randomized_eigsh(
            a,
            n_components=num_components,
            n_iter=iterated_power,
            random_state=random_state,
            selection="module",
        )

    # If solver is 'dense', use standard scipy.linalg.eigh
    # Note: subset_by_index specifies the indices of smallest/largest to return
    index = (_num_features(a) - num_components, _num_features(a) - 1)
    return la.eigh(a, b, subset_by_index=index)


# Postprocess eignevalues and eigenvectors


def _postprocess_eigencomponents(eigenvalues, eigenvectors, steps, num_components=None, remove_zero_eig=False):
    """Postprocess the eigenvalues and eigenvectors after eigendecomposition.
    Args:
        eigenvalues (array-like): The eigenvalues of the kernel matrix.
        eigenvectors (array-like): The eigenvectors of the kernel matrix.
        steps (list): The steps to perform in postprocessing, including:
                        - "remove_significant_negative_eigenvalues"
                        - "check_psd_eigenvalues"
                        - "svd_flip"
                        - "sort_eigencomponents"
                        - "keep_positive_eigenvalues"
        num_components (int, optional): The number of components to keep. If None, all components are kept.
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues.
    Returns:
        tuple: `(eigenvalues, eigenvectors)` after postprocessing.
    """
    for step in steps:
        if step == "remove_significant_negative_eigenvalues":
            eigenvalues = _remove_significant_negative_eigenvalues(eigenvalues)
        if step == "check_psd_eigenvalues":
            eigenvalues = _check_psd_eigenvalues(eigenvalues)
        if step == "svd_flip":
            eigenvectors, _ = svd_flip(eigenvectors, None)
        if step == "sort_eigencomponents":
            eigenvalues, eigenvectors = _sort_eigencomponents(eigenvalues, eigenvectors)
        if step == "keep_positive_eigenvalues":
            eigenvalues, eigenvectors = _keep_positive_eigenvalues(
                eigenvalues, eigenvectors, num_components, remove_zero_eig
            )

    return eigenvalues, eigenvectors


def _sort_eigencomponents(eigenvalues, eigenvectors):
    """Sort the eigenvalues and eigenvectors in descending order.
    Args:
        eigenvalues (array-like): The eigenvalues of the kernel matrix.
        eigenvectors (array-like): The eigenvectors of the kernel matrix.
    Returns:
        tuple: `(eigenvalues, eigenvectors)` sorted in descending order.
    """
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors


def _keep_positive_eigenvalues(eigenvalues, eigenvectors, num_components=None, remove_zero_eig=False):
    """Keep only the positive eigenvalues and their corresponding eigenvectors. If `num_components is not None`,
    or `remove_zero_eig=False`, the non-positive eigenvalues will not be removed.
    Args:
        eigenvalues (array-like): The eigenvalues of the kernel matrix.
        eigenvectors (array-like): The eigenvectors of the kernel matrix.
        num_components (int, optional): The number of components to keep. If None, all components are kept.
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues.
    Returns:
        tuple: `(eigenvalues, eigenvectors)` with only positive eigenvalues and their corresponding eigenvectors.
    """
    if remove_zero_eig or num_components is None:
        pos_mask = eigenvalues > 0
        eigenvectors = eigenvectors[:, pos_mask]
        eigenvalues = eigenvalues[pos_mask]

    return eigenvalues, eigenvectors


def _remove_significant_negative_eigenvalues(lambdas):
    """Remove significant negative eigenvalues from the eigenvalues array.
    Args:
        lambdas (array-like): The eigenvalues of the kernel matrix.
    Returns:
        array-like: The eigenvalues with significant negative values set to zero.
    """
    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6

    lambdas = np.real(lambdas)
    max_eig = lambdas.max()

    significant_neg_eigvals_index = lambdas < -significant_neg_ratio * max_eig
    significant_neg_eigvals_index &= lambdas < -significant_neg_value

    lambdas[significant_neg_eigvals_index] = 0

    return lambdas


def _scale_eigenvectors(eigenvalues, eigenvectors):
    """Scale the eigenvectors by the square root of the eigenvalues.
    Args:
        eigenvalues (array-like): The eigenvalues of the kernel matrix.
        eigenvectors (array-like): The eigenvectors of the kernel matrix.
    Returns:
        array-like: The scaled eigenvectors.
    """
    s = np.sqrt(eigenvalues)

    non_zeros = np.flatnonzero(s)
    eigenvectors_ = np.zeros_like(eigenvectors)
    eigenvectors_[:, non_zeros] = eigenvectors[:, non_zeros] / s[non_zeros]

    return eigenvectors_


class MPCA(BaseEstimator, TransformerMixin):
    """MPCA implementation compatible with scikit-learn

    Args:
        var_ratio (float, optional): Percentage of variance explained (between 0 and 1). Defaults to 0.97.
        max_iter (int, optional): Maximum number of iteration. Defaults to 1.
        vectorize (bool): Whether return the transformed/projected tensor in vector. Defaults to False.
        n_components (int): Number of components to keep. Applies only when vectorize=True. Defaults to None.

    Attributes:
        proj_mats (list of arrays): A list of transposed projection matrices, shapes (P_1, I_1), ...,
            (P_N, I_N), where P_1, ..., P_N are output tensor shape for each sample.
        idx_order (array-like): The ordering index of projected (and vectorized) features in decreasing variance.
        mean_ (array-like): Per-feature empirical mean, estimated from the training set, shape (I_1, I_2, ..., I_N).
        shape_in (tuple): Input tensor shapes, i.e. (I_1, I_2, ..., I_N).
        shape_out (tuple): Output tensor shapes, i.e. (P_1, P_2, ..., P_N).

    Reference:
        Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear Principal Component Analysis of
        Tensor Objects", IEEE Transactions on Neural Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial
        Matlab implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.

    Examples:
        >>> import numpy as np
        >>> from kale.embed.factorization import MPCA
        >>> x = np.random.random((40, 20, 25, 20))
        >>> x.shape
        (40, 20, 25, 20)
        >>> mpca = MPCA()
        >>> x_projected = mpca.fit_transform(x)
        >>> x_projected.shape
        (40, 18, 23, 18)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 7452)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 50)
        >>> x_rec = mpca.inverse_transform(x_projected)
        >>> x_rec.shape
        (40, 20, 25, 20)
    """

    def __init__(self, var_ratio=0.97, max_iter=1, vectorize=False, n_components=None):
        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            msg = "Number of max iterations must be a positive integer but given %s" % max_iter
            logging.error(msg)
            raise ValueError(msg)
        self.proj_mats = []
        self.vectorize = vectorize
        self.n_components = n_components

    def fit(self, x, y=None):
        """Fit the model with input training data x.

        Args
            x (array-like tensor): Input data, shape (n_samples, I_1, I_2, ..., I_N), where n_samples is the number of
                samples, I_1, I_2, ..., I_N are the dimensions of corresponding mode (1, 2, ..., N), respectively.
            y (None): Ignored variable.

        Returns:
            self (object). Returns the instance itself.
        """
        self._fit(x)
        return self

    def _fit(self, x):
        """Solve MPCA"""

        shape_ = x.shape  # shape of input data
        n_dims = x.ndim

        self.shape_in = shape_[1:]
        self.mean_ = np.mean(x, axis=0)
        x = x - self.mean_

        # init
        shape_out = ()
        proj_mats = []

        # get the output tensor shape based on the cumulative distribution of eigen values for each mode
        for i in range(1, n_dims):
            mode_data_mat = unfold(x, mode=i)
            singular_vec_left, singular_val, singular_vec_right = la.svd(mode_data_mat, full_matrices=False)
            eig_values = np.square(singular_val)
            idx_sorted = (-1 * eig_values).argsort()
            cum = eig_values[idx_sorted]
            tot_var = np.sum(cum)

            for j in range(1, cum.shape[0] + 1):
                if np.sum(cum[:j]) / tot_var > self.var_ratio:
                    shape_out += (j,)
                    break
            proj_mats.append(singular_vec_left[:, idx_sorted][:, : shape_out[i - 1]].T)

        # set n_components to the maximum n_features if it is None
        if self.n_components is None:
            self.n_components = int(np.prod(shape_out))

        for i_iter in range(self.max_iter):
            for i in range(1, n_dims):  # ith mode
                x_projected = multi_mode_dot(
                    x,
                    [proj_mats[m] for m in range(n_dims - 1) if m != i - 1],
                    modes=[m for m in range(1, n_dims) if m != i],
                )
                mode_data_mat = unfold(x_projected, i)

                singular_vec_left, singular_val, singular_vec_right = la.svd(mode_data_mat, full_matrices=False)
                eig_values = np.square(singular_val)
                idx_sorted = (-1 * eig_values).argsort()
                proj_mats[i - 1] = (singular_vec_left[:, idx_sorted][:, : shape_out[i - 1]]).T

        x_projected = multi_mode_dot(x, proj_mats, modes=[m for m in range(1, n_dims)])
        x_proj_unfold = unfold(x_projected, mode=0)  # unfold the tensor projection to shape (n_samples, n_features)
        # x_proj_cov = np.diag(np.dot(x_proj_unfold.T, x_proj_unfold))  # covariance of unfolded features
        x_proj_cov = np.sum(np.multiply(x_proj_unfold.T, x_proj_unfold.T), axis=1)  # memory saving computing covariance
        idx_order = (-1 * x_proj_cov).argsort()

        self.proj_mats = proj_mats
        self.idx_order = idx_order
        self.shape_out = shape_out
        self.n_dims = n_dims

        return self

    def transform(self, x):
        """Perform dimension reduction on x

        Args:
            x (array-like tensor): Data to perform dimension reduction, shape (n_samples, I_1, I_2, ..., I_N).

        Returns:
            array-like tensor:
                Projected data in lower dimension, shape (n_samples, P_1, P_2, ..., P_N) if self.vectorize==False.
                If self.vectorize==True, features will be sorted based on their explained variance ratio, shape
                (n_samples, P_1 * P_2 * ... * P_N) if self.n_components is None, and shape (n_samples, n_components)
                if self.n_component is a valid integer.
        """
        # reshape x to shape (1, I_1, I_2, ..., I_N) if x in shape (I_1, I_2, ..., I_N), i.e. n_samples = 1
        if x.ndim == self.n_dims - 1:
            x = x.reshape((1,) + x.shape)
        _check_tensor_dim_shape(x, self.n_dims, self.shape_in)
        x = x - self.mean_

        # projected tensor in lower dimensions
        x_projected = multi_mode_dot(x, self.proj_mats, modes=[m for m in range(1, self.n_dims)])

        if self.vectorize:
            x_projected = unfold(x_projected, mode=0)
            x_projected = x_projected[:, self.idx_order]
            if isinstance(self.n_components, int):
                n_features = int(np.prod(self.shape_out))
                if self.n_components > n_features:
                    self.n_components = n_features
                    warn_msg = "n_components exceeds the maximum number, all features will be returned."
                    logging.warning(warn_msg)
                    warnings.warn(warn_msg)
                x_projected = x_projected[:, : self.n_components]

        return x_projected

    def inverse_transform(self, x):
        """Reconstruct projected data to the original shape and add the estimated mean back

        Args:
            x (array-like tensor): Data to be reconstructed, shape (n_samples, P_1, P_2, ..., P_N), if
                self.vectorize == False, where P_1, P_2, ..., P_N are the reduced dimensions of corresponding
                mode (1, 2, ..., N), respectively. If self.vectorize == True, shape (n_samples, self.n_components)
                or shape (n_samples, P_1 * P_2 * ... * P_N).

        Returns:
            array-like tensor:
                Reconstructed tensor in original shape, shape (n_samples, I_1, I_2, ..., I_N)
        """
        # reshape x to tensor in shape (n_samples, self.shape_out) if x has been unfolded
        if x.ndim <= 2:
            if x.ndim == 1:
                # reshape x to a 2D matrix (1, n_components) if x in shape (n_components,)
                x = x.reshape((1, -1))
            n_samples = x.shape[0]
            n_features = x.shape[1]
            if n_features <= np.prod(self.shape_out):
                x_ = np.zeros((n_samples, np.prod(self.shape_out)))
                x_[:, self.idx_order[:n_features]] = x[:]
            else:
                msg = "Feature dimension exceeds the shape upper limit."
                logging.error(msg)
                raise ValueError(msg)

            x = fold(x_, mode=0, shape=((n_samples,) + self.shape_out))

        x_rec = multi_mode_dot(x, self.proj_mats, modes=[m for m in range(1, self.n_dims)], transpose=True)

        x_rec = x_rec + self.mean_

        return x_rec


class BaseKernelDomainAdapter(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for kernel domain adaptation methods. Extendable to support different
    kernel-based domain adaptation methods (e.g., MIDA, TCA, SCA).

    Args:
        num_components (int, optional): Number of components to keep. If None, all components are kept. Defaults to None.
        ignore_y (bool, optional): Whether to ignore the target variable `y` during fitting. Defaults to False.
        augment (str, optional): Whether to augment the input data with factors. Can be "pre" (prepend factors),
            "post" (append factors), or None (no augmentation). Defaults to None.
        kernel (str or callable, optional): Kernel function to use. Can be "linear", "rbf", "poly", "sigmoid", or a callable. Defaults to "linear".
        gamma (float, optional): Kernel coefficient for "rbf", "poly", and "sigmoid" kernels. If None, defaults to 1 / num_features. Defaults to None.
        degree (int, optional): Degree of the polynomial kernel. Ignored by other kernels. Defaults to 3.
        coef0 (float, optional): Independent term in the polynomial and sigmoid kernels. Ignored by other kernels. Defaults to 1.
        kernel_params (dict, optional): Additional parameters for the kernel function. Defaults to None.
        alpha (float, optional): Regularization parameter for the kernel. Defaults to 1.0.
        fit_inverse_transform (bool, optional): Whether to fit the inverse transform for reconstruction. Defaults to False.
        eigen_solver (str, optional): Eigenvalue solver to use. Can be "auto", "dense", "arpack", or "randomized". Defaults to "auto".
        tol (float, optional): Tolerance for convergence of the eigenvalue solver. Defaults to 0.
        max_iter (int, optional): Maximum number of iterations for the eigenvalue solver. If None, no limit is applied. Defaults to None.
        iterated_power (int or str, optional): Number of iterations for the randomized solver. Can be an integer or "auto". Defaults to "auto".
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues during postprocessing. Defaults to False.
        scale_components (bool, optional): Whether to scale the components by the square root of their eigenvalues. Defaults to False.
        random_state (int, np.random.RandomState, or None, optional): Random seed for reproducibility. Defaults to None.
        copy (bool, optional): Whether to copy the input data during validation. Defaults to True.
        num_jobs (int or None, optional): Number of jobs to run in parallel for pairwise kernel computations. Defaults to None.
    """

    _parameter_constraints: dict = {
        "num_components": [Interval(Integral, 1, None, closed="left"), None],
        "ignore_y": ["boolean"],
        "augment": [StrOptions({"pre", "post"}), None],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS) | {"precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "fit_inverse_transform": ["boolean"],
        "eigen_solver": [StrOptions({"auto", "dense", "arpack", "randomized"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "iterated_power": [
            Interval(Integral, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "remove_zero_eig": ["boolean"],
        "scale_components": ["boolean"],
        "random_state": ["random_state"],
        "copy": ["boolean"],
        "num_jobs": [None, Integral],
    }

    _eigen_preprocess_steps = [
        "remove_significant_negative_eigenvalues",
        "check_psd_eigenvalues",
        "svd_flip",
        "sort_eigencomponents",
        "keep_positive_eigenvalues",
    ]

    def __init__(
        self,
        num_components=None,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        num_jobs=None,
    ):
        # Truncation parameters
        self.num_components = num_components

        # Supervision parameters
        self.ignore_y = ignore_y

        # Kernel parameters
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.copy = copy
        self.num_jobs = num_jobs

        # Eigendecomposition parameters
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

        # Transform parameters
        self.scale_components = scale_components

        # Inverse transform parameters
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform

        # Additional adaptation parameters
        self.augment = augment

        # init additional attributes
        self.classes_ = None
        self.gamma_ = None
        self.x_fit_ = None
        self._factor_validator = None
        self._centerer = None

    def _fit_inverse_transform(self, x_transformed, x):
        if hasattr(x, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for sparse matrices!")

        n_samples = x_transformed.shape[0]
        k_x = self._get_kernel(x_transformed)
        k_x.flat[:: n_samples + 1] += self.alpha
        self.dual_coef_ = la.solve(k_x, x, assume_a="pos", overwrite_a=True)
        self.x_transformed_fit_ = x_transformed

    @property
    def _n_features_out(self):
        """Number of features out after transformation."""

        # The property name can't be changed as it is used
        # by scikit-learn's core module to validate.
        return self.eigenvalues_.shape[0]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _more_tags(self):
        return {
            "_xfail_checks": {"check_transformer_n_iter": "Follows similar implementation to KernelPCA."},
        }

    def _get_kernel(self, x, y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(x, y, metric=self.kernel, filter_params=True, n_jobs=self.num_jobs, **params)

    @property
    def orig_coef_(self):
        """Coefficients projected to the original feature space
        with shape (num_components, num_features).
        """
        check_is_fitted(self)
        if self.kernel != "linear":
            raise NotImplementedError("Available only when `kernel=True`.")
        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)
        return safe_sparse_dot(w.T, self.x_fit_)

    def _fit_transform_in_place(self, k_x):
        """Fit the model to the kernel matrix `k_x` and perform eigendecomposition in place.
        Args:
            k_x (array-like): The kernel matrix.
        """
        eigenvalues, eigenvectors = _eigendecompose(
            k_x,
            self.num_components,
            self.eigen_solver,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            iterated_power=self.iterated_power,
        )

        eigenvalues, eigenvectors = _postprocess_eigencomponents(
            eigenvalues,
            eigenvectors,
            self._eigen_preprocess_steps,
            num_components=self.num_components,
            remove_zero_eig=self.remove_zero_eig,
        )

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

    def _make_objective_kernel(self, k_x, y, factors):
        """Create the objective kernel for the eigendecomposition.
        Args:
            k_x (array-like): The kernel matrix.
            y (array-like): The target variable (binary or multiclass classification label) with shape (num_samples).
            factors (array-like): The factors for adaptation with shape (num_samples, num_factors).
                                Please preprocess the factors before domain adaptation
                                (e.g. one-hot encode domain, gender, or standardize age).
        Returns:
            array-like: The objective kernel.
        """
        return k_x

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y=None, group_labels=None, **fit_params):
        """Fit the model to the data `x` and target variable `y`.
        Args:
            x (array-like): The input data with shape (num_samples, num_features).
            y (array-like, optional): The target variable (binary or multiclass classification label) with shape (num_samples).
                                    Set -1 for unknown labels for semi-supervised MIDA. Default is None.
            group_labels (array-like, optional): Categorical variables representing domain or grouping factors with shape
                (num_samples, num_factors). Preprocessing (e.g., one-hot encode domain, gender, or age groups)
                must be applied in advance. Default is None.
            **fit_params: Additional parameters for fitting.
        Returns:
            self: The fitted model.
        """

        # Data validation for x, y, and factors
        check_params = {
            "accept_sparse": False if self.fit_inverse_transform else "csr",
            "copy": self.copy,
        }
        if y is None or self.ignore_y:
            x = validate_data(self, x, **check_params)
            y_ohe = np.zeros((_num_samples(x), 1), dtype=x.dtype)
        else:
            x, y = validate_data(self, x, y, **check_params)
            y_type = type_of_target(y)

            if y_type not in ["binary", "multiclass"]:
                raise ValueError(f"y should be a 'binary' or 'multiclass' target. Got '{y_type}' instead.")

            drop = (-1,) if np.any(y == -1) else None
            ohe = OneHotEncoder(sparse_output=False, drop=drop)
            y_ohe = ohe.fit_transform(np.expand_dims(y, 1))

            self.classes_ = ohe.categories_[0]

        if group_labels is None:
            raise ValueError(f"Group labels must be provided for `{self.__class__.__name__}` during `fit`.")

        # k_objective workaround to validate the group_labels' shape
        factor_validator = FunctionTransformer()
        factor_validator.fit(group_labels, x)
        group_labels = factor_validator.transform(group_labels)

        # Append the factors/phenotypes to the input data if augment=True
        x_aug = x
        if self.augment is not None:
            self._factor_validator = factor_validator

        if self.augment == "pre":
            x_aug = np.hstack((x, group_labels))

        # To avoid having duplicate variables, x_fit_ cannot be renamed
        # to x_fit_ as it is predefined in KernelPCA's implementation
        self.x_fit_ = x_aug
        self.gamma_ = 1 / _num_features(x) if self.gamma is None else self.gamma
        self._centerer = KernelCenterer()

        k_x = self._get_kernel(self.x_fit_)
        k_x = self._centerer.fit_transform(k_x)

        k_objective = self._make_objective_kernel(k_x, y_ohe, group_labels)

        # Fit the transformation and inverse transformation for the kernel matrix
        self._fit_transform_in_place(k_objective)
        if self.fit_inverse_transform:
            x_transformed = self.transform(x, group_labels)
            self._fit_inverse_transform(x_transformed, x)

        return self

    def transform(self, x, group_labels=None):
        """Transform the input data `x` to factor-independent feature space using the fitted
        domain adapter.
        Args:
            x (array-like): The input data with shape (num_samples, num_features).
            group_labels (array-like, optional): Categorical variables representing domain or grouping factors with
                shape (num_samples, num_factors). Preprocessing (e.g., one-hot encode domain, gender, or age groups)
                must be applied in advance. Default is None.
        Returns:
            array-like: The transformed data with shape (num_samples, num_components).
        """
        check_is_fitted(self)
        accept_sparse = False if self.fit_inverse_transform else "csr"
        x = validate_data(self, x, accept_sparse=accept_sparse, reset=False)

        if group_labels is None and self.augment in {"pre", "post"}:
            raise ValueError("Factors must be provided for transform when `augment` is 'pre' or 'post'.")

        if self.augment == "pre":
            x = np.hstack((x, group_labels))

        k_x = self._get_kernel(x, self.x_fit_)
        k_x = self._centerer.transform(k_x)

        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)

        z = safe_sparse_dot(k_x, w)

        if self.augment == "post":
            z = np.hstack((z, group_labels))

        return z

    def inverse_transform(self, z):
        """Inverse transform the transformed data `z` back to the original space.
        Args:
            z (array-like): The transformed data with shape (num_samples, num_components).
        Returns:
            array-like: The inverse transformed data with shape (num_samples, num_features)
                        or (num_samples, num_features + num_factors) if `augment=True`.
        """
        check_is_fitted(self)
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        k_z = self._get_kernel(z, self.x_transformed_fit_)
        return safe_sparse_dot(k_z, self.dual_coef_)

    def fit_transform(self, x, y=None, group_labels=None, **fit_params):
        """Fit the model to the data `x` and target variable `y` and remove
        the effect of `factors`, and transform `x`.
        Args:
            x (array-like): The input data with shape (num_samples, num_features).
            y (array-like, optional): The target variable (binary or multiclass classification label) with shape (num_samples).
                                    Set -1 for unknown labels for semi-supervised MIDA. Default is None.
            group_labels (array-like, optional): Categorical variables representing domain or grouping factors with
                shape (num_samples, num_factors). Preprocessing (e.g., one-hot encode domain, gender, or age groups)
                must be applied in advance. Default is None.
            **fit_params: Additional parameters for fitting.
        Returns:
            array-like: The transformed data.
        """
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            self.fit(x, group_labels=group_labels, **fit_params)
        else:
            # fit method of arity 2 (supervised transformation)
            self.fit(x, y, group_labels, **fit_params)

        return self.transform(x, group_labels)


class MIDA(BaseKernelDomainAdapter):
    """Maximum Independent Domain Adaptation (MIDA).
    A kernel-based domain adaptation method that uses removes the effect of
    factors/covariates from the data by learning a feature space derived from maximizing
    Hilbert-Schmidt independence criterion (HSIC).

    To prevent label leakage, please set the label for the target indices to -1.

    Args:
        num_components (int, optional): Number of components to keep. If None, all components are kept.
        mu (float, optional): L2 kernel regularization coefficient. Default is 1.0.
        eta (float, optional): Class-dependency regularization coefficient. Default is 1.0.
        ignore_y (bool, optional): Whether to ignore the target variable `y` during fitting. Default is False.
        augment (str, optional): Whether to augment the input data with factors. Can be "pre" (prepend factors),
            "post" (append factors), or None (no augmentation). Defaults to None.
        kernel (str, optional): Kernel type to be used. Default is 'linear'.
        gamma (float, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Default is None.
        degree (int, optional): Degree of the polynomial kernel. Default is 3.
        coef0 (float, optional): Independent term in the polynomial and sigmoid kernels. Default is 1.
        kernel_params (dict, optional): Additional kernel parameters. Default is None.
        alpha (float, optional): Regularization parameter. Default is 1.0.
        fit_inverse_transform (bool, optional): Whether to fit the inverse transform. Default is False.
        eigen_solver (str, optional): Eigendecomposition solver to use. Default is 'auto'.
        tol (float, optional): Tolerance for convergence. Default is 0.
        max_iter (int, optional): Maximum number of iterations for the solver. Default is None.
        iterated_power (int or str, optional): Number of iterations for randomized solver. Default is 'auto'.
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues. Default is False.
        scale_components (bool, optional): Whether to scale the components. Default is False.
        random_state (int or np.random.RandomState, optional): Random seed for reproducibility. Default is None.
        copy (bool, optional): Whether to copy the input data. Default is True.
        num_jobs (int, optional): Number of jobs to run in parallel for joblib.Parallel. Default is None.
    References:
        [1] Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace using domain features and
            independence maximization. IEEE transactions on cybernetics, 48(1), pp.288-299.
    Examples:
        >>> import numpy as np
        >>> from kale.embed.factorization import MIDA
        >>> # Generate random synthetic data
        >>> x_source = np.random.normal(loc=5, scale=1, size=(20, 40))
        >>> x_target = np.random.normal(loc=-5, scale=1, size=(20, 40))
        >>> y = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
        >>> # Concatenate source and target data
        >>> x = np.vstack((x_source, x_target))
        >>> target_indices = np.arange(20, 40)
        >>> # Mask the target indices with -1
        >>> y[target_indices] = -1
        >>> # Create factors (e.g., one-hot encoded domain labels)
        >>> factors = np.concatenate((np.zeros((20, 1)), np.ones((20, 1))), axis=0)
        >>> mida = MIDA()
        >>> x_projected = mida.fit_transform(x, y, group_labels=factors)
        >>> x_projected.shape
        (40, 18)
    """

    _parameter_constraints: dict = {
        **BaseKernelDomainAdapter._parameter_constraints,
        "mu": [Interval(Real, 0, None, closed="neither")],
        "eta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        num_components=None,
        mu=1.0,
        eta=1.0,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        num_jobs=None,
    ):
        # L2 kernel regularization parameter
        self.mu = mu
        # Class dependency regularization parameter
        self.eta = eta

        # Kernel and Eigendecomposition parameters
        super().__init__(
            num_components=num_components,
            ignore_y=ignore_y,
            augment=augment,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            num_jobs=num_jobs,
        )

    def _make_objective_kernel(self, k_x, y, group_labels):
        # equivalent to `H` in the original paper
        h = _centering_kernel(_num_features(k_x), k_x.dtype)
        # linear kernel used for the label and factors
        k_y = pairwise_kernels(y, n_jobs=self.num_jobs)
        k_f = pairwise_kernels(group_labels, n_jobs=self.num_jobs)

        centerer = KernelCenterer()
        k_y = centerer.fit_transform(k_y)
        k_f = centerer.fit_transform(k_f)

        k_objective = multi_dot((k_x, self.mu * h + self.eta * k_y - k_f, k_x))

        return k_objective

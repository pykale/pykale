# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Python implementation of Multilinear Principal Component Analysis (MPCA)

Reference:
    Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear Principal Component Analysis of Tensor
    Objects", IEEE Transactions on Neural Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial Matlab
    implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.
"""
import logging
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# import tensorly as tl
from tensorly.base import fold, unfold
from tensorly.tenalg import multi_mode_dot


def _check_n_dim(x, n_dims):
    """Raise error if the number of dimensions of the input data is not consistent with the expected value.

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


class MPCA(BaseEstimator, TransformerMixin):
    """MPCA implementation compatible with sickit-learn

    Args:
        var_ratio (float, optional): Percentage of variance explained (between 0 and 1). Defaults to 0.97.
        max_iter (int, optional): Maximum number of iteration. Defaults to 1.
        return_vector (bool): Whether ruturn the transformed/projected tensor in vector. Defaults to False.
        n_components (int): Number of components to keep. Applies only when return_vector=True. Defaults to None.

    Attributes:
        proj_mats (list of array-like): A list of transposed projection matrices, shapes (P_1, I_1), ...,
            (P_N, I_N), where P_1, ..., P_N are output tensor shape for each sample.
        idx_order (array-like): The ordering index of projected (and vectorised) features in decreasing variance.
        mean_ (array-like): Per-feature empirical mean, estimated from the training set, shape (I_1, I_2, ..., I_N).
        shape_in (tuple): Input tensor shapes, i.e. (I_1, I_2, ..., I_N).
        shape_out (tuple): Output tensor shapes, i.e. (P_1, P_2, ..., P_N).
    Examples:
        >>> import numpy as np
        >>> from kale.embed import MPCA
        >>> x = np.random.random((40, 20, 25, 20))
        >>> x.shape
        (40, 20, 25, 20)
        >>> mpca = MPCA(variance_explained=0.9)
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

    def __init__(self, var_ratio=0.97, max_iter=1, return_vector=False, n_components=None):
        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            msg = "Number of max iterations must be a positive integer but given %s" % max_iter
            logging.error(msg)
            raise ValueError(msg)
        self.proj_mats = []
        self.return_vector = return_vector
        self.n_components = n_components

    def fit(self, x, y=None):
        """Fit the model with input training data X.

        Args
            X (array-like tensor): Input data, shape (n_samples, I_1, I_2, ..., I_N), where n_samples is the number of
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
        n_samples = shape_[0]  # number of samples
        n_dims = x.ndim

        self.shape_in = shape_[1:]
        self.mean_ = np.mean(x, axis=0)
        x = x - self.mean_

        # init
        phi = dict()
        shape_out = ()
        proj_mats = []

        for i in range(1, n_dims):
            for j in range(n_samples):
                # unfold the j_th tensor along the i_th mode
                x_ji = unfold(x[j, :], mode=i - 1)
                if i not in phi:
                    phi[i] = 0
                phi[i] = phi[i] + np.dot(x_ji, x_ji.T)

        # get the output tensor shape based on the cumulative distribution of eigenvalues for each mode
        for i in range(1, n_dims):
            eig_values, eig_vectors = np.linalg.eig(phi[i])
            idx_sorted = (-1 * eig_values).argsort()
            cum = eig_values[idx_sorted]
            var_tot = np.sum(cum)

            for j in range(1, cum.shape[0] + 1):
                if np.sum(cum[:j]) / var_tot > self.var_ratio:
                    shape_out += (j,)
                    break
            proj_mats.append(eig_vectors[:, idx_sorted][:, : shape_out[i - 1]].T)

        # set n_components to the maximum n_features if it is None
        if self.n_components is None:
            self.n_components = int(np.prod(shape_out))

        for i_iter in range(self.max_iter):
            phi = dict()
            for i in range(1, n_dims):  # ith mode
                if i not in phi:
                    phi[i] = 0
                for j in range(n_samples):
                    xj = multi_mode_dot(
                        x[j, :],  # jth tensor/sample
                        [proj_mats[m] for m in range(n_dims - 1) if m != i - 1],
                        modes=[m for m in range(n_dims - 1) if m != i - 1],
                    )
                    xj_unfold = unfold(xj, i - 1)
                    phi[i] = np.dot(xj_unfold, xj_unfold.T) + phi[i]

                eig_values, eig_vectors = np.linalg.eig(phi[i])
                idx_sorted = (-1 * eig_values).argsort()
                proj_mats[i - 1] = (eig_vectors[:, idx_sorted][:, : shape_out[i - 1]]).T

        x_projected = multi_mode_dot(x, proj_mats,
                                     modes=[m for m in range(1, n_dims)])
        x_proj_unfold = unfold(x_projected, mode=0)  # unfold the tensor projection to shape (n_samples, n_features)
        # x_proj_cov = np.diag(np.dot(x_proj_unfold.T, x_proj_unfold))  # covariance of unfolded features
        x_proj_cov = np.sum(np.multiply(x_proj_unfold, x_proj_unfold), axis=1)  # memory saving computing covariance
        idx_order = (-1 * x_proj_cov).argsort()

        self.proj_mats = proj_mats
        self.idx_order = idx_order
        self.shape_out = shape_out
        self.n_dims = n_dims

        return self

    def transform(self, x):
        """Perform dimension reduction on X

        Args:
            x (array-like tensor): Data to perform dimension reduction, shape (n_samples, I_1, I_2, ..., I_N).

        Returns:
            array-like tensor:
                Projected data in lower dimension, shape (n_samples, P_1, P_2, ..., P_N) if self.return_vector==False.
                If self.return_vector==True, features will be sorted based on their explained variance ratio, shape
                (n_samples, P_1 * P_2 * ... * P_N) if self.n_components is None, and shape (n_samples, n_components)
                if self.n_component is a valid integer.
        """
        _check_tensor_dim_shape(x, self.n_dims, self.shape_in)
        x = x - self.mean_

        # projected tensor in lower dimensions
        x_projected = multi_mode_dot(
            x, self.proj_mats,
            modes=[m for m in range(1, self.n_dims)])

        if self.return_vector:
            x_projected = unfold(x_projected, mode=0)
            x_projected = x_projected[:, self.idx_order]
            if isinstance(self.n_components, int):
                if self.n_components > np.prod(self.shape_out):
                    self.n_components = np.prod(self.shape_out)
                    warn_msg = "n_components exceeds the maximum number, all features will be returned."
                    logging.warning(warn_msg)
                    warnings.warn(warn_msg)
                x_projected = x_projected[:, : self.n_components]

        return x_projected

    def inverse_transform(self, x):
        """Reconstruct projected data to the original shape and add the estimated mean back

        Args:
            x (array-like tensor): Data to be reconstructed, shape (n_samples, P_1, P_2, ..., P_N), if
                self.return_vector == False, where P_1, P_2, ..., P_N are the reduced dimensions of of corresponding
                mode (1, 2, ..., N), respectively. If self.return_vector == True, shape (n_samples, self.n_components)
                or shape (n_samples, P_1 * P_2 * ... * P_N).

        Returns:
            array-like tensor:
                Reconstructed tensor in original shape, shape (n_samples, I_1, I_2, ..., I_N)
        """
        # reshape X to tensor in shape (n_samples, self.shape_out) if X is vectorised
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

        x_rec = multi_mode_dot(x, self.proj_mats,
                               modes=[m for m in range(1, self.n_dims)],
                               transpose=True)

        x_rec = x_rec + self.mean_

        return x_rec

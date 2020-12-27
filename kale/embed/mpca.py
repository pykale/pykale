# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Python implementation of Multilinear Principal Component Analysis (MPCA)
Includeing implementaion as a scikit-learn object and an independent function.

Reference:
    Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear
    Principal Component Analysis of Tensor Objects", IEEE Transactions on Neural 
    Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial Matlab
    implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.
"""

import sys
import numpy as np
# import tensorly as tl
from tensorly.base import unfold
from tensorly.tenalg import multi_mode_dot
from sklearn.base import BaseEstimator, TransformerMixin


def check_ndim(X, ndim):
    """Check if the order/mode of the input data is consistent with the given number of order/mode

    Args:
        X (ndarray): input data
        ndim (int): number of order/mode expected

    Returns:
        boolean value: True if the order of X is consistent with ndim, False vise versa
    """

    return X.ndim == ndim


def check_shape(X, shape_):
    """Check if the shape (excluding the last mode) of the input data is consistent with the given shape

    Args:
        X (ndarray): input data
        shape_: shape expected

    Returns:
        boolean value: True if the shape of X is consistent with shape_, False vise versa
    """

    return X.shape[:-1] == shape_


def _check_dim_shape(X, ndim, shape_):
    """Check if the order/mode of the input data is consistent with the given number of order/mode

    Args:
        X (ndarray): input data
        ndim (int): number of order/mode expected
        shape_: shape expected

    """
    if not check_ndim(X, ndim) or not check_shape(X, shape_):
        print("The given data should be consistent with the order and shape of training data")
        sys.exit()


class MPCA(BaseEstimator, TransformerMixin):
    """MPCA implementation compatible with sickit-learn

    Args:
        var_ratio (float, optional): Percentage of variance explained
            (between 0 and 1). Defaults to 0.97.
        max_iter (int, optional): Maximum number of iteration. Defaults to 1.

    Attributes:
        proj_mats (list): A list of transposed projection matrices.
        idx_order (ndarray): The ordering index of projected (and vectorised) features in decreasing variance.
        mean_ (ndarray): Per-feature empirical mean, estimated from the training set, shape (I_1, I_2, ..., I_N).
        shape_in (tuple): Input tensor shapes, i.e. (I_1, I_2, ..., I_N).
        shape_out (tuple): Output tensor shapes, i.e. (P_1, P_2, ..., P_N).
    Examples:
        >>> import numpy as np
        >>> from kale.embed.mpca import MPCA
        >>> X = np.random.random((20, 25, 20, 40))
        >>> X.shape
        (20, 25, 20, 40)
        >>> mpca = MPCA(variance_explained=0.9)
        >>> X_transformed = mpca.fit_transform(X)
        >>> X_transformed.shape
        (18, 23, 18, 40)
        >>> X_transformed = mpca.transform(X, vectorise=Ture)
        >>> X_transformed.shape
        (40, 7452)
        >>> X_transformed = mpca.transform(X, vectorise=Ture, n_components=50)
        >>> X_transformed.shape
        (40, 50)
        >>> X_inverse = mpca.inverse_transform(X_transformed)
        >>> X_inverse.shape
        (20, 25, 20, 40)
    """
    def __init__(self, var_ratio=0.97, max_iter=1):
        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            print('Number of max iterations must be an integer and greater than 0')
            sys.exit()
        self.proj_mats = []

    def fit(self, X, y=None):
        """Fit the model with input training data X.

        Args
            X (ndarray): Input data, shape (I_1, I_2, ..., I_N, n_samples), where n_samples
                is the number of samples, I_1, I_2, ..., I_N are the dimensions of 
                corresponding mode (1, 2, ..., N), respectively.
            y (None): Ignored variable.

        Returns:
            self (object). Returns the instance itself.
        """
        self._fit(X)
        return self

    def _fit(self, X):
        """Solve MPCA"""

        shape_ = X.shape  # shape of input data
        n_spl = shape_[-1]  # number of samples
        self.ndim = X.ndim
        self.shape_in = shape_[:-1]
        self.mean_ = np.mean(X, axis=-1)
        X_ = np.zeros(X.shape)
        # init
        Phi = dict()
        Vs = dict()  # dictionary of eigenvectors for all modes/orders
        lambdas = dict()  # dictionary of eigenvalues for all modes/orders
        # dictionary of cumulative distribution of eigenvalues for all modes/orders
        cums = dict()  
        proj_mats = []
        shape_out = ()
        for i in range(n_spl):
            X_[..., i] = X[..., i] - self.mean_
            for j in range(self.ndim - 1):
                X_i = unfold(X_[..., i], mode=j)
                if j not in Phi:
                    Phi[j] = 0
                Phi[j] = Phi[j] + np.dot(X_i, X_i.T)

        for i in range(self.ndim - 1):
            eig_vals, eig_vecs = np.linalg.eig(Phi[i])
            idx_sorted = eig_vals.argsort()[::-1]
            Vs[i] = eig_vecs[:, idx_sorted]
            cum = eig_vals[idx_sorted]
            var_tot = np.sum(cum)

            for j in range(cum.shape[0]):
                if np.sum(cum[:j]) / var_tot > self.var_ratio:
                    shape_out += (j + 1,)
                    break
            cums[i] = cum
            # if Vs[i][0] / sum_ > self.var_ratio:
            #     cums[i] = 1
            # else:
            #     for j in range(Vs[i].shape[0] - 1, 0, -1):
            #         if np.sum(Vs[i][j:]) / sum_ > (1 - self.var_ratio):
            #             cums[i] = j + 1
            #             break
            proj_mats.append(Vs[i][:, :shape_out[i]].T)

        for i_iter in range(self.max_iter):
            Phi = dict()
            for i in range(self.ndim - 1):  # ith mode
                if i not in Phi:
                    Phi[i] = 0
                for j in range(n_spl):
                    X_j = X_[..., j]  # jth tensor/sample
                    Xj_ = multi_mode_dot(X_j, [proj_mats[m] for m in range(self.ndim - 1) if m != i],
                                         modes=[m for m in range(self.ndim - 1) if m != i])
                    Xj_unfold = unfold(Xj_, i)
                    Phi[i] = np.dot(Xj_unfold, Xj_unfold.T) + Phi[i]

                eig_vals, eig_vecs = np.linalg.eig(Phi[i])
                idx_sorted = eig_vals.argsort()[::-1]
                lambdas[i] = eig_vals[idx_sorted]
                proj_mats[i] = eig_vecs[:, idx_sorted]
                proj_mats[i] = (proj_mats[i][:, :shape_out[i]]).T

        x_transformed = multi_mode_dot(X, proj_mats, modes=[m for m in range(self.ndim - 1)])
        x_trans_vec = unfold(x_transformed, mode=-1)  # vectorise the transformed features
        # diagonal of the covariance of vectorised features
        x_trans_cov_diag = np.diag(np.dot(x_trans_vec, x_trans_vec.T))
        idx_order = x_trans_cov_diag.argsort()[::-1]

        self.proj_mats = proj_mats
        self.idx_order = idx_order
        self.shape_out = shape_out

        return self

    def transform(self, X, vectorise=False, n_components=None):
        """Perform dimension reduction on X

        Args:
            X (ndarray): Data to perform dimension reduction, shape (I_1, I_2, ..., I_N, n_samples).
            vectorise (bool): Whether vectorise the transformed tensor. Defaults to Flase.
            n_components (int): Number of components to keep. Applies only when vectorise=True. 
                Defaults to None.

        Returns:
            ndarray: Transformed data, shape (P_1, P_2, ..., P_N, n_samples).
        """
        _check_dim_shape(X, self.ndim, self.shape_in)
        n_spl = X.shape[-1]
        for i in range(n_spl):
            X[..., i] = X[..., i] - self.mean_
        
        X_transformed = multi_mode_dot(X, self.proj_mats, modes=[m for m in range(self.ndim - 1)])

        if vectorise:
            X_transformed = unfold(X_transformed, mode=-1)
            if isinstance(n_components, int) and n_components<=np.prod(self.shape_out):
                X_transformed = X_transformed[:, self.idx_order]
                X_transformed = X_transformed[:, :n_components]

        return X_transformed

    def inverse_transform(self, X):
        """Transform data in the shape of reduced dimension 
        back to the original shape

        Args:
            X (ndarray): New data, shape (P_1, P_2, ..., P_N, n_samples), where P_1, P_2, ..., P_N 
                are the reduced dimensions of of corresponding mode (1, 2, ..., N), respectively.

        Returns:
            ndarray: Data in original shape, shape (I_1, I_2, ..., I_N, n_samples)
        """
        _check_dim_shape(X, self.ndim, self.shape_out)
        return multi_mode_dot(X, self.proj_mats, modes=[m for m in range(self.ndim - 1)], transpose=True)


# The following implementation kept for a tensorly function
# def MPCA_(X, var_explained=0.97, max_iter=1):
#     """MPCA implementation as an independent function
#
#     Args:
#         X (ndarray): training data, shape (P_1, P_2, ..., P_N, n_samples)
#         var_explained (float, optional): ration of variance to keep (between 0 and 1). Defaults to 0.97.
#         max_iter (int, optional): max number of iteration. Defaults to 1.
#
#     Returns:
#         list: a list of transposed projection matrices
#
#     Examples:
#         >>> import numpy as np
#         >>> from tensorly.tenalg import multi_mode_dot
#         >>> from kale.embed.mpca import MPCA_
#         >>> X = np.random.random((20, 25, 20, 40))
#         >>> X.shape
#         (20, 25, 20, 40)
#         >>> proj_mats = MPCA_(X, variance_explained=0.9)
#         >>> X_transformed = multi_mode_dot(X, proj_mats, modes=[0, 1, 2])
#         >>> X_transformed.shape
#         (18, 23, 18, 40)
#     """
#     dim_in = X.shape
#     n_spl = dim_in[-1]
#     n_dim = X.ndim
#     # Is = dim_in[:-1]
#     Xmean = np.mean(X, axis=-1)
#     X_ = np.zeros(X.shape)
#     # init
#     Phi = dict()
#     Us = dict()  # eigenvectors
#     Vs = dict()  # eigenvalues
#     cums = dict()  # cumulative distribution of eigenvalues
#     proj_mats = []
#     for i in range(n_spl):
#         X_[..., i] = X[..., i] - Xmean
#         for j in range(n_dim-1):
#             X_i = unfold(X_[..., i], mode=j)
#             if j not in Phi:
#                 Phi[j] = 0
#             Phi[j] = Phi[j] + np.dot(X_i, X_i.T)
#
#     for i in range(n_dim-1):
#         eig_vals, eig_vecs = np.linalg.eig(Phi[i])
#         idx_sorted = eig_vals.argsort()[::-1]
#         Us[i] = eig_vecs[:, idx_sorted]
#         Vs[i] = eig_vals[idx_sorted]
#         # sum_ = np.sum(Vs[i])
#         # for j in range(Vs[i].shape[0] - 1, 0, -1):
#         #     if np.sum(Vs[i][j:]) / sum_ > (1 - variance_explained):
#         #         cums[i] = j + 1
#         #         break
#
#         var_tot = np.sum(Vs[i])
#
#         for j in range(Vs[i].shape[0]):
#             if np.sum(Vs[i][:j]) / var_tot > var_explained:
#                 cums[i] = j + 1
#                 break
#
#         proj_mats.append(Us[i][:, :cums[i]].T)
#
#     for i_iter in range(max_iter):
#         Phi = dict()
#         for i in range(n_dim-1):
#             # dim_in_ = dim_in[i]
#             if i not in Phi:
#                 Phi[i] = 0
#             for j in range(n_spl):
#                 X_i = X_[..., j]
#                 Xi_ = multi_mode_dot(X_i, [proj_mats[m] for m in range(n_dim-1) if m != i],
#                                      modes=[m for m in range(n_dim) if m != i])
#                 tXi = unfold(Xi_, i)
#                 Phi[i] = np.dot(tXi, tXi.T) + Phi[i]
#
#             eig_vals, eig_vecs = np.linalg.eig(Phi[i])
#             idx_sorted = eig_vals.argsort()[::-1]
#             proj_mats[i] = eig_vecs[:, idx_sorted]
#             proj_mats[i] = proj_mats[i][:, :cums[i]].T
#
#     return proj_mats

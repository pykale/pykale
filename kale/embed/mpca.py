# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk
# =============================================================================

"""Python implementation of Multilinear Principal Component Analysis (MPCA)
Includeing implementaion as a scikit-learn class and an independent function

Ref: Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear
    Principal Component Analysis of Tensor Objects", IEEE Transactions on Neural 
    Networks, Vol. 19, No. 1, Page: 18-39, January 2008.
"""

import numpy as np
import tensorly as tl
from tensorly.base import fold, unfold
from tensorly.tenalg import mode_dot, multi_mode_dot
from sklearn.base import BaseEstimator, TransformerMixin


class MPCA(BaseEstimator, TransformerMixin):
    def __init__(self, variance_explained=0.97, max_iter=1):
        """MPCA implementation compatible with sickit-learn

        Args:
            variance_explained (float, optional): ration of variance to keep 
                (between 0 and 1). Defaults to 0.97.
            max_iter (int, optional): max number of iteration. Defaults to 1.

        Attributes:
            tPs: list of projection matrices

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
        >>> X_inverse = mpca.inverse_transform(X_transformed)
        >>> X_inverse.shape
        (20, 25, 20, 40)
        """
        self.var_exp = variance_explained
        self.max_iter = max_iter
        # self.Xmean = None
        # self.n_dim = None
        self.tPs = []

    def fit(self, X):
        """Fit the model with X

        Args:
            X (ndarray): input data, shape (I1, I2, ..., n_samples), where
            n_samples is the number of samples, I1, I2 ... are the dimensions 
            of corresponding mode (1, 2, ...), respectively.

        Returns:
            self
        """
        dim_in = X.shape
        n_spl = dim_in[-1]
        self.n_dim = X.ndim
        self.Xmean = np.mean(X, axis=-1)
        X_ = np.zeros(X.shape)
        # init
        Phi = dict()
        Us = dict()  # eigenvectors
        Vs = dict()  # eigenvalues
        cums = dict()  # cumulative distribution of eigenvalues
        tPs = []
        for i in range(n_spl):
            X_[..., i] = X[..., i] - self.Xmean
            for j in range(self.n_dim - 1):
                X_i = unfold(X_[..., i], mode=j)
                if j not in Phi:
                    Phi[j] = 0
                Phi[j] = Phi[j] + np.dot(X_i, X_i.T)

        for i in range(self.n_dim - 1):
            eig_vals, eig_vecs = np.linalg.eig(Phi[i])
            idx_sorted = eig_vals.argsort()[::-1]
            Us[i] = eig_vecs[:, idx_sorted]
            Vs[i] = eig_vals[idx_sorted]
            sum_ = np.sum(Vs[i])
            for j in range(Vs[i].shape[0] - 1, 0, -1):
                if np.sum(Vs[i][j:]) / sum_ > (1 - self.var_exp):
                    cums[i] = j + 1
                    break
            tPs.append(Us[i][:, :cums[i]].T)

        for i_iter in range(self.max_iter):
            Phi = dict()
            for i in range(self.n_dim - 1):
                if i not in Phi:
                    Phi[i] = 0
                for j in range(n_spl):
                    X_i = X_[..., j]
                    Xi_ = multi_mode_dot(X_i, [tPs[m] for m in range(self.n_dim - 1) if m != i],
                                         modes=[m for m in range(self.n_dim - 1) if m != i])
                    tXi = unfold(Xi_, i)
                    Phi[i] = np.dot(tXi, tXi.T) + Phi[i]

                eig_vals, eig_vecs = np.linalg.eig(Phi[i])
                idx_sorted = eig_vals.argsort()[::-1]
                tPs[i] = eig_vecs[:, idx_sorted]
                tPs[i] = tPs[i][:, :cums[i]].T
        self.tPs = tPs

        return self

    def transform(self, X):
        """Perform dimension reduction on X

        Args:
            X (ndarray): shape (I1, I2, ..., n_samples)

        Returns:
            ndarray: Transformed data, shape (i1, i2, ..., n_samples)
        """
        n_spl = X.shape[-1]
        for i in range(n_spl):
            X[..., i] = X[..., i] - self.Xmean
        return multi_mode_dot(X, self.tPs, modes=[m for m in range(self.n_dim - 1)])

    def inverse_transform(self, X):
        """Transform data in the shape of reduced dimension 
        back to the original shape

        Args:
            X (ndarray): shape (i1, i2, ..., n_samples), where i1, i2, ... 
            are the reduced dimensions of of corresponding mode
            (1, 2, ...), respectively.

        Returns:
            ndarray: Data in original shape, shape (I1, I2, ..., n_samples)
        """
        
        return multi_mode_dot(X, self.tPs, modes=[m for m in range(self.n_dim - 1)], transpose=True)


def MPCA_(X, variance_explained=0.97, max_iter=1):
    """MPCA implementation as a function

    Args:
        X (ndarray): training data, shape (I1, I2, ..., n_samples)
        variance_explained (float, optional): ration of variance to keep (between 0 and 1). Defaults to 0.97.
        max_iter (int, optional): max number of iteration. Defaults to 1.

    Returns:
        list: tPs: list of projection matrices
    
    Examples:
    >>> import numpy as np
    >>> from tensorly.tenalg import multi_mode_dot
    >>> from kale.embed.mpca import MPCA_
    >>> X = np.random.random((20, 25, 20, 40))
    >>> X.shape
    (20, 25, 20, 40)
    >>> tPs = MPCA_(X, variance_explained=0.9)
    >>> X_transformed = multi_mode_dot(X, tPs, modes=[0, 1, 2])
    >>> X_transformed.shape
    (18, 23, 18, 40)
    """
    dim_in = X.shape
    n_spl = dim_in[-1]
    n_dim = X.ndim
    # Is = dim_in[:-1]
    Xmean = np.mean(X, axis=-1)
    X_ = np.zeros(X.shape)
    # init
    Phi = dict()
    Us = dict()  # eigenvectors
    Vs = dict()  # eigenvalues
    cums = dict()  # cumulative distribution of eigenvalues
    tPs = []
    for i in range(n_spl):
        X_[..., i] = X[..., i] - Xmean
        for j in range(n_dim-1):
            X_i = unfold(X_[..., i], mode=j)
            if j not in Phi:
                Phi[j] = 0
            Phi[j] = Phi[j] + np.dot(X_i, X_i.T)

    for i in range(n_dim-1):
        eig_vals, eig_vecs = np.linalg.eig(Phi[i])
        idx_sorted = eig_vals.argsort()[::-1]
        Us[i] = eig_vecs[:, idx_sorted]
        Vs[i] = eig_vals[idx_sorted]
        sum_ = np.sum(Vs[i])
        for j in range(Vs[i].shape[0] - 1, 0, -1):
            if np.sum(Vs[i][j:]) / sum_ > (1 - variance_explained):
                cums[i] = j + 1
                break
        tPs.append(Us[i][:, :cums[i]].T)

    for i_iter in range(max_iter):
        Phi = dict()
        for i in range(n_dim-1):
            # dim_in_ = dim_in[i]
            if i not in Phi:
                Phi[i] = 0
            for j in range(n_spl):
                X_i = X_[..., j]
                Xi_ = multi_mode_dot(X_i, [tPs[m] for m in range(n_dim-1) if m != i],
                                     modes=[m for m in range(n_dim) if m != i])
                tXi = unfold(Xi_, i)
                Phi[i] = np.dot(tXi, tXi.T) + Phi[i]

            eig_vals, eig_vecs = np.linalg.eig(Phi[i])
            idx_sorted = eig_vals.argsort()[::-1]
            tPs[i] = eig_vecs[:, idx_sorted]
            tPs[i] = tPs[i][:, :cums[i]].T

    return tPs

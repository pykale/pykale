# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk
#
# Python implement of Multilinear Principal Component Analysis (MPCA)
#
# Ref: Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear
# Principal Component Analysis of Tensor Objects", IEEE Transactions on Neural 
# Networks, Vol. 19, No. 1, Page: 18-39, January 2008.
# =============================================================================
import numpy as np
import tensorly as tl
from tensorly.base import fold, unfold
from tensorly.tenalg import mode_dot, multi_mode_dot
from sklearn.base import BaseEstimator, TransformerMixin


class MPCA(BaseEstimator, TransformerMixin):
    def __init__(self, variance_explained=0.97, max_iter=1):
        """
        Implementation compatible with sickit-learn
        ----------
        Parameter:
            variance_explained: ration of variance to keep (between 0 and 1)
            max_iter: max number of iteration, integer
        ----------
        Attributes:
            tPs: list of projection matrices
        """
        self.var_exp = variance_explained
        self.max_iter = max_iter
        # self.Xmean = None
        # self.n_dim = None
        self.tPs = []

    def fit(self, X):
        """
        Parameter:
            X: array-like, ndarray of shape (dim1, dim2, ..., n_samples)
        ----------
        Return:
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
        """
        Parameter:
            X: array-like, ndarray of shape (dim1, dim2, ..., n_samples)
        ----------
        Return:
            Transformed data
        """
        n_spl = X.shape[-1]
        for i in range(n_spl):
            X[..., i] = X[..., i] - self.Xmean
        return multi_mode_dot(X, self.tPs, modes=[m for m in range(self.n_dim - 1)])


def MPCA_(X, variance_explained=0.97, max_iter=1):
    """
    Parameter:
        X: array-like, ndarray of shape (dim1, dim2, ..., n_samples)
        variance_explained: ration of variance to keep (between 0 and 1)
        max_iter: max number of iteration
    ----------
    Return:
        tPs: list of projection matrices
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



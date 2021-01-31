# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Multilinear Principal Component Analysis (MPCA) implementation in Pytorch

Reference:
    Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear
    Principal Component Analysis of Tensor Objects", IEEE Transactions on Neural
    Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial Matlab
    implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.
"""
# import pytorch_lightning as pl
import warnings

import tensorly as tl
import torch
import torch.nn as nn

# from tensorly import fold, unfold
from tensorly.tenalg import multi_mode_dot

from ._mpca import _check_dim_shape

tl.set_backend("pytorch")


class MPCA(nn.Module):
    """MPCA implementation compatible with pytorch

    Args:
        var_ratio (float, optional): Percentage of variance explained
            (between 0 and 1). Defaults to 0.97.
        max_iter (int, optional): Maximum number of iteration. Defaults to 1.


    Attributes:
        factors (list of array-like): A list of transposed projection matrices, shapes (P_1, I_1), ...,
            (P_N, I_N), where P_1, ..., P_N are output tensor shape for each sample.
        idx_order (tensor): The ordering index of projected (and vectorised) features in decreasing variance.
        # mean_ (tensor): Per-feature empirical mean, estimated from the training set, shape (I_1, I_2, ..., I_N).
        shape_in (tuple): Input tensor shapes, i.e. (I_1, I_2, ..., I_N).
        shape_out (tuple): Output tensor shapes, i.e. (P_1, P_2, ..., P_N).
    """

    def __init__(self, var_ratio=0.97, max_iter=1):
        super().__init__()

        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            raise ValueError(
                "Number of max iterations must be an integer and greater " "than 0, but given %s" % max_iter
            )
        self.factors = []
        self.shape_in = ()
        self.shape_out = ()
        # self.mean_ = None
        self.idx_order = None
        self.n_dim = None

    def _init_model(self, x):
        """Init model parameters for x"""
        shape_ = x.shape  # shape of input data
        n_spl = shape_[0]  # number of samples
        n_dim = x.ndim

        # self.mean_ = torch.mean(x, 0)
        # x_ = x - self.mean_

        phi = dict()

        self.shape_in = shape_[1:]
        shape_out = ()

        for i in range(1, n_dim):
            for j in range(n_spl):
                # unfold the j_th tensor along the i_th mode
                x_ji = tl.unfold(x[j, :], mode=i - 1)
                if i not in phi:
                    phi[i] = 0
                phi[i] = phi[i] + torch.matmul(x_ji, x_ji.T)

        for i in range(1, n_dim):
            eig_values, eig_vectors = torch.eig(phi[i], eigenvectors=True)
            eig_values = eig_values[:, 0]
            idx_sorted = (-1 * eig_values).argsort()
            # ev_sorted[i] = eig_vectors[:, idx_sorted]
            cum = eig_values[idx_sorted]
            var_tot = sum(cum)

            for j in range(1, cum.shape[0] + 1):
                if sum(cum[:j]) / var_tot > self.var_ratio:
                    shape_out += (j,)
                    break
            # cums[i] = cum
            self.factors.append(eig_vectors[:, idx_sorted][:, : shape_out[i - 1]].T)

        all_comps = int(torch.prod(torch.Tensor(shape_out)))
        self.idx_order = range(all_comps)
        self.shape_out = shape_out
        self.n_dim = n_dim

    def _update_model(self, x):
        """update model parameters for x"""

        n_spl = x.shape[0]  # number of samples

        for i_iter in range(self.max_iter):
            phi = dict()
            for i in range(1, self.n_dim):  # ith mode
                if i not in phi:
                    phi[i] = 0
                for j in range(n_spl):
                    x_j = x[j, :]  # jth tensor/sample
                    xj_ = multi_mode_dot(
                        x_j,
                        [self.factors[m] for m in range(self.n_dim - 1) if m != i - 1],
                        modes=[m for m in range(self.n_dim - 1) if m != i - 1],
                    )
                    xj_unfold = tl.unfold(xj_, i - 1)
                    phi[i] = torch.matmul(xj_unfold, xj_unfold.T) + phi[i]

                eig_values, eig_vectors = torch.eig(phi[i], eigenvectors=True)
                eig_values = eig_values[:, 0]
                idx_sorted = (-1 * eig_values).argsort()
                # lambdas[i] = eig_values[idx_sorted]
                self.factors[i - 1] = (eig_vectors[:, idx_sorted][:, : self.shape_out[i - 1]]).T

        tensor_pc_vec = self.forward(x, return_vector=True, sort_by_var=False)
        # return_vector the tensor principal components
        # diagonal of the covariance of vectorised features
        tensor_pc_cov_diag = torch.diag(torch.matmul(tensor_pc_vec.T, tensor_pc_vec))
        idx_order = (-1 * tensor_pc_cov_diag).argsort()

        self.idx_order = idx_order

        return self

    def training_step(self, train_batch):
        """Update the model parameters for a training batch"""
        x, y = train_batch
        if len(self.factors) == 0:
            self._init_model(x)
        self._update_model(x)
        return self

    def forward(self, x, return_vector=False, sort_by_var=True, n_components=None):
        """Perform dimension reduction on x

        Args:
            x (tensor):
            # reduce_mean (bool): Whether reduce the mean estimated from training data. Defaults to True.
            return_vector (bool): Whether vectorize the tensor principal components. Defaults to False.
            sort_by_var (bool): Whether sort features by variance in descending order. Defaults to True,
                applies only when return_vector=True.
            n_components (int): Number of components to keep. Applies only when return_vector=True.
                Defaults to None.

        Returns:
            tensor
                tensor principal components, shape (n_samples, P_1, P_2, ..., P_N) if return_vector==False.
                If return_vector==True, features will be vectorized and sorted based on their explained
                variance ratio, shape (n_samples, P_1 * P_2 * ... * P_N) if self.n_components is None,
                and shape (n_samples, n_components) if self.n_component is not None.
        """
        _check_dim_shape(x, self.n_dim, self.shape_in)

        # if reduce_mean:
        #     x -= self.mean_

        # tensor principal components
        tensor_pc = multi_mode_dot(x, self.factors, modes=[m for m in range(1, self.n_dim)])

        if return_vector:
            tensor_pc = tl.unfold(tensor_pc, mode=0)
            if sort_by_var:
                tensor_pc = tensor_pc[:, self.idx_order]
            if isinstance(n_components, int):
                all_comp = int(torch.prod(torch.Tensor(self.shape_out)))
                if n_components > all_comp:
                    warnings.warn(
                        "n_components can not exceed the multiplication of output "
                        "tensor shapes, the maximum number of components will be applied."
                    )
                    n_components = all_comp
                if not sort_by_var:
                    tensor_pc = tensor_pc[:, self.idx_order]
                tensor_pc = tensor_pc[:, :n_components]

        return tensor_pc

    def restore(self, x):
        """Transform data in the shape of reduced dimension back to the original shape

        Args:
            x (tensor): Data to be restored, shape (n_samples, P_1, P_2, ..., P_N), where
                P_1, P_2, ..., P_N are the reduced dimensions of of corresponding mode
                (1, 2, ..., N), respectively. For tensors have been vectorized,
                shape (n_samples, self.n_components) or (n_samples, P_1 * P_2 * ... * P_N).
            # add_mean (bool): Whether add the mean estimated from the training set to the output.
            #     Defaults to False.

        Returns:
            tensor:
                Reconstructed tensor in original shape, shape (n_samples, I_1, I_2, ..., I_N).
        """
        if x.ndim <= 2:
            if x.ndim == 1:
                x = x.reshape((1, -1))
            n_spl = x.shape[0]
            n_feat = x.shape[1]
            all_comp = int(torch.prod(torch.Tensor(self.shape_out)))
            if n_feat <= all_comp:
                x_ = torch.zeros((n_spl, all_comp))
                x_[:, : self.idx_order[:n_feat]] = x[:]
            else:
                raise ValueError("Feature dimension exceeds the shape upper limit.")

            x = tl.fold(x_, mode=0, shape=((n_spl,) + self.shape_out))

        # restored tensor
        x_rst = multi_mode_dot(x, self.factors, modes=[m for m in range(1, self.n_dim)], transpose=True)

        # if add_mean:
        #     x_rec += self.mean_

        return x_rst

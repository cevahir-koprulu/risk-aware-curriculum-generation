import scipy.linalg as scpla
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from deep_sprl.util.torch import get_weights, set_weights, to_float_tensor
import copy
from deep_sprl.util.torch_distribution import TorchDistribution


class GaussianTorchDistribution(TorchDistribution):

    def __init__(self, mu, chol_flat, use_cuda, dtype=torch.float32):
        super().__init__(use_cuda)
        self._dim = mu.shape[0]

        self._mu = nn.Parameter(torch.as_tensor(mu, dtype=dtype), requires_grad=True)
        self._chol_flat = nn.Parameter(torch.as_tensor(chol_flat, dtype=dtype), requires_grad=True)

        self.distribution_t = MultivariateNormal(self._mu, scale_tril=self.to_tril_matrix(self._chol_flat, self._dim))

    def __copy__(self):
        return GaussianTorchDistribution(self._mu, self._chol_flat, self.use_cuda)

    def __deepcopy__(self, memodict=None):
        return GaussianTorchDistribution(copy.deepcopy(self._mu), copy.deepcopy(self._chol_flat), self.use_cuda)

    @staticmethod
    def to_tril_matrix(chol_flat, dim):
        if isinstance(chol_flat, np.ndarray):
            chol = np.zeros((dim, dim))
            exp_fun = np.exp
        else:
            chol = torch.zeros((dim, dim), dtype=chol_flat.dtype)
            exp_fun = torch.exp

        d1, d2 = np.diag_indices(dim)
        chol[d1, d2] += exp_fun(chol_flat[0: dim])
        ld1, ld2 = np.tril_indices(dim, k=-1)
        chol[ld1, ld2] += chol_flat[dim:]

        return chol

    @staticmethod
    def flatten_matrix(mat, tril=False):
        if not tril:
            mat = scpla.cholesky(mat, lower=True)

        dim = mat.shape[0]
        d1, d2 = np.diag_indices(dim)
        ld1, ld2 = np.tril_indices(dim, k=-1)

        return np.concatenate((np.log(mat[d1, d2]), mat[ld1, ld2]))

    def entropy_t(self):
        return self.distribution_t.entropy()

    def mean_t(self):
        return self.distribution_t.mean

    def log_pdf_t(self, x):
        return self.distribution_t.log_prob(x)

    def sample(self, *args, **kwargs):
        return self.distribution_t.rsample(*args, **kwargs)

    def covariance_matrix(self):
        return self.distribution_t.covariance_matrix.detach().numpy()

    def set_weights(self, weights):
        set_weights([self._mu], weights[0:self._dim], self._use_cuda)
        set_weights([self._chol_flat], weights[self._dim:], self._use_cuda)
        # This is important - otherwise the changes will not be reflected!
        self.distribution_t = MultivariateNormal(self._mu, scale_tril=self.to_tril_matrix(self._chol_flat, self._dim))

    @staticmethod
    def from_weights(dim, weights, use_cuda=False, dtype=torch.float32):
        mu = weights[0: dim]
        chol_flat = weights[dim:]
        return GaussianTorchDistribution(mu, chol_flat, use_cuda=use_cuda, dtype=dtype)

    def get_weights(self):
        mu_weights = get_weights([self._mu])
        chol_flat_weights = get_weights([self._chol_flat])

        return np.concatenate([mu_weights, chol_flat_weights])

    def parameters(self):
        return [self._mu, self._chol_flat]

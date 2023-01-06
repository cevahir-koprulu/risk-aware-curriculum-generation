import copy
import torch
import numpy as np
from deep_sprl.aux_teachers.cem.cem import CEM
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution

class CEMGaussian(CEM):
    '''CEM for Gaussian distribution'''
    def __init__(self, *args, **kwargs):
        super(CEMGaussian, self).__init__(*args, **kwargs)

    def sample(self):
        sample_ok = False
        count = 0
        while not sample_ok and count < 100:
            sample = self.sample_dist[-1].sample().detach().numpy()
            sample_ok = np.all(self.data_bounds[0] <= sample) and (np.all(sample <= self.data_bounds[1]))
            count += 1

        if sample_ok:
            return sample
        else:
            mu = self.sample_dist[-1].mean()
            # Why uniform sampling? Because if we sample 100 times outside of the allowed
            if np.all(self.data_bounds[0] <= mu) and (np.all(mu <= self.data_bounds[1])):
                return np.random.uniform(self.data_bounds[0], self.data_bounds[1])
            else:
                return np.clip(sample, self.data_bounds[0], self.data_bounds[1])

    def pdf(self, x, use_original_dist=False):
        if use_original_dist:
            log_p = self.original_dist.log_pdf(x)
        else:
            log_p = self.sample_dist[-1].log_pdf(x)
        return np.exp(log_p)

    def update_sample_distribution(self, samples, weights):
        weights_scaled = weights/np.mean(weights)
        samples_weighted = (weights_scaled*samples.T).T
        mean = np.mean(samples_weighted, axis=0)
        s = np.zeros((self.context_dim, self.context_dim))
        for i in range(samples.shape[0]):
            s += np.matmul(np.sqrt(weights_scaled[i]) * (samples_weighted[i, :] - mean).reshape(-1, 1),
                           np.sqrt(weights_scaled[i]) * (samples_weighted[i, :] - mean).reshape(-1, 1).T)
        var = (1./samples.shape[0])*s

        # Create the initial context distribution
        if isinstance(var, np.ndarray):
            flat_chol = GaussianTorchDistribution.flatten_matrix(var, tril=False)
        else:
            flat_chol = GaussianTorchDistribution.flatten_matrix(var * np.eye(self.context_dim),
                                                                      tril=False)
        dist = GaussianTorchDistribution(mean, flat_chol, use_cuda=False, dtype=torch.float64)
        self.sample_dist.append(dist)

    def update_original_distribution(self, dist_params):
        mean, var = dist_params
        self.context_dim = mean.shape[0]
        # Create the initial context distribution
        if isinstance(var, np.ndarray):
            flat_chol = GaussianTorchDistribution.flatten_matrix(var, tril=False)
        else:
            flat_chol = GaussianTorchDistribution.flatten_matrix(var * np.eye(self.context_dim),
                                                                      tril=False)
        dist = GaussianTorchDistribution(mean, flat_chol, use_cuda=False, dtype=torch.float64)
        self.original_dist = dist
        self.original_dists.append(copy.copy(dist))

    def update_ref_alpha(self):
        num_update = len(self.sample_dist)-1
        ref_alpha_new = self.risk_level_scheduler(update_no=num_update)
        self.ref_alpha = ref_alpha_new

    def get_sample_dist(self):
        mean = self.sample_dist[-1].mean()
        std = np.sqrt(np.diag(self.sample_dist[-1].covariance_matrix()))
        return (mean, std)

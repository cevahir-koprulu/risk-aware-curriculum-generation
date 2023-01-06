import copy
import torch
import numpy as np
from deep_sprl.aux_teachers.cem.cem import CEM
from deep_sprl.util.cauchy_torch_distribution import CauchyTorchDistribution

class CEMCauchy(CEM):
    '''CEM for Cauchy distribution'''
    def __init__(self, *args, **kwargs):
        super(CEMCauchy, self).__init__(*args, **kwargs)

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
            loc = self.sample_dist[-1].mean()
            # Why uniform sampling? Because if we sample 100 times outside of the allowed
            if np.all(self.data_bounds[0] <= loc) and (np.all(loc <= self.data_bounds[1])):
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
        dim = samples.shape[1]
        loc_prev = self.sample_dist[-1].mean()
        scale_prev = self.sample_dist[-1].covariance_matrix()
        h = np.array([(1+((samples[s_i, :] - loc_prev) @ np.linalg.inv(scale_prev) @ (samples[s_i, :] - loc_prev)))
                      for s_i in range(samples.shape[0])])
        weights_new = weights / h
        weights_scaled = weights_new/np.mean(weights_new)
        samples_weighted = (weights_scaled*samples.T).T
        loc = np.mean(samples_weighted, axis=0)
        scale = np.zeros((dim, dim))
        for i in range(samples.shape[0]):
            scale += (dim + 1) * weights_new[i] / np.sum(weights) * \
                     np.matmul((samples[i, :] - loc).reshape(-1, 1),
                               (samples[i, :] - loc).reshape(-1, 1).T)

        # Create the initial context distribution
        if isinstance(scale, np.ndarray):
            flat_chol = CauchyTorchDistribution.flatten_matrix(scale, tril=False)
        else:
            flat_chol = CauchyTorchDistribution.flatten_matrix(scale * np.eye(dim),
                                                                      tril=False)
        dist = CauchyTorchDistribution(loc, flat_chol, use_cuda=False, dtype=torch.float64)
        self.sample_dist.append(dist)

    def update_original_distribution(self, dist_params):
        loc, scale = dist_params
        self.context_dim = loc.shape[0]
        # Create the initial context distribution
        if isinstance(scale, np.ndarray):
            flat_chol = CauchyTorchDistribution.flatten_matrix(scale, tril=False)
        else:
            flat_chol = CauchyTorchDistribution.flatten_matrix(scale * np.eye(self.context_dim),
                                                                      tril=False)
        dist = CauchyTorchDistribution(loc, flat_chol, use_cuda=False, dtype=torch.float64)
        self.original_dist = dist
        self.original_dists.append(copy.copy(dist))

    def update_ref_alpha(self):
        num_update = len(self.sample_dist)-1
        ref_alpha_new = self.risk_level_scheduler(update_no=num_update)
        self.ref_alpha = ref_alpha_new

    def get_sample_dist(self):
        loc = self.sample_dist[-1].mean()
        scale = np.sqrt(np.diag(self.sample_dist[-1].covariance_matrix()))
        return (loc, scale)

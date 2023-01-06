'''
This module implements the Cross Entropy Method (CEM) for sampling of low quantiles.

The user should have a stochastic process P(score; theta), whose distribution
depends on the parameter theta (the module was originally developed to sample
"difficult" conditions in reinforcement learning, so theta was the parameters
of the environment, and the score was the return of the episode).

The basic CEM class is abstract: any use of it requires inheritance with
implementation of the following methods:
- do_sample(dist): returning a sample from the distribution represented by dist.
- pdf(x, dist): the probability of x under the distribution dist.
- update_sample_distribution(samples, weights): updating dist given new samples.
- likelihood_ratio(x) (optional): implemented by default as
                                  pdf(x, orig_dist) / pdf(x, curr_dist).
                                  the user may provide a more efficient or stable
                                  implementation according to the underlying
                                  family of distributions.
Note that dist is an object that represents the distribution, and its type is
up to the user. A standard type may be a list of distribution parameters.

Examples for inheritance from CEM are provided below - see CEM_Ber, CEM_Beta.
A simple usage example is provided in the bottom of this file - see __main__.

Module structure:
CEM:
    sample(): return a sample from the current distribution, along with a weight
              corresponding to the likelihood-ratio wrt the original distribution.
        do_sample(curr_dist): do the sampling.              [IMPLEMENTED BY USER]
        get_weight(x): calculate the LR weight.
            likelihood_ratio(x).
                pdf(x, dist).                               [IMPLEMENTED BY USER]
    update(score):
        update list of scores.
        if there're enough samples and it's time to update the distribution:
            select_samples().
            update_sample_distribution(samples, weights).   [IMPLEMENTED BY USER]
CEM_Ber: an inherited class implementing CEM for a 1D Bernoulli distribution.
CEM_Beta: an inherited class implementing CEM for a 1D Beta distribution.

Written by Ido Greenberg, 2022.
'''

import numpy as np
import pickle as pkl
import copy, warnings


class CEM:
    def __init__(self, dist_params, target_log_likelihood, risk_level_scheduler, data_bounds,
                 batch_size=0, ref_mode='train', ref_alpha=0.05, n_orig_per_batch=0.2,
                 internal_alpha=0.2, force_min_samples=True, w_clip=5, title='cem'):
        self.title = title
        # An object defining the original distribution to sample from.
        # This can be any object (e.g., list of distribution parameters),
        # depending on the implementation of the inherited class.
        self.context_dim = None
        self.original_dist = None
        self.original_dists = []  # n_batches
        self.update_original_distribution(dist_params)

        # Extension
        self.target_log_likelihood = target_log_likelihood
        self.risk_level_scheduler = risk_level_scheduler
        self.data_bounds = data_bounds

        # Number of samples to draw before updating distribution.
        # 0 is interpreted as infinity.
        self.batch_size = batch_size

        # Clip the likelihood-ratio weights to the range [1/w_clip, w_clip].
        # If None or 0 - no clipping is done.
        self.w_clip = w_clip

        # How to use reference scores to determine the threshold for the
        # samples selected for distribution update?
        # - 'none': ignore reference scores.
        # - 'train': every batch, draw the first n=n_orig_per_batch samples
        #            from the original distribution instead of the updated one.
        #            then use quantile(batch_scores[:n]; ref_alpha).
        # - 'valid': use quantile(external_validation_scores; ref_alpha).
        #            in this case, update_ref_scores() must be called to
        #            feed reference scores before updating the distribution.
        # In CVaR optimization, ref_alpha would typically correspond to
        # the CVaR risk level.
        self.ref_mode = ref_mode
        self.ref_alpha = ref_alpha

        # Number of samples to draw every batch from the original distribution
        # instead of the updated one. Either integer or ratio in (0,1).
        self.n_orig_per_batch = n_orig_per_batch
        if 0<self.n_orig_per_batch<1:
            self.n_orig_per_batch = int(self.n_orig_per_batch*self.batch_size)
        if self.batch_size < self.n_orig_per_batch:
            warnings.warn(f'samples per batch = {self.batch_size} < '
                          f'{self.n_orig_per_batch} = original-dist samples per batch')
            self.n_orig_per_batch = self.batch_size

        active_train_mode = self.ref_mode == 'train' and self.batch_size
        if active_train_mode and self.n_orig_per_batch < 1:
            raise ValueError('"train" reference mode must come with a positive'
                             'number of original-distribution samples per batch.')

        # In a distribution update, use from the current batch at least
        # internal_alpha * (batch_size - n_orig_per_batch)
        # samples. internal_alpha should be in the range (0,1).
        self.internal_alpha = internal_alpha
        if active_train_mode:
            self.internal_alpha *= 1 - self.n_orig_per_batch / self.batch_size
        # If multiple scores R equal the alpha quantile q, then the selected
        #  R<q samples may be strictly fewer than internal_alpha*batch_size.
        #  If force_min_samples==True, we fill in the missing entries from
        #  samples with R==q.
        self.force_min_samples = force_min_samples

        # State
        self.ref_scores = None

        # Data
        self.sample_dist = [copy.copy(self.original_dist)]  # n_batches
        self.sampled_data = [[]]  # n_batches x batch_size
        self.weights = [[]]  # n_batches x batch_size
        self.scores = [[]]  # n_batches x batch_size
        self.ref_quantile = []  # n_batches
        self.internal_quantile = []  # n_batches
        self.selected_samples = [[]]  # n_batches x batch_size

    def save(self, log_dir):
        filename = f'{log_dir}/{self.title}.cem'
        obj = (
            self.title, self.original_dist, self.batch_size, self.w_clip,
            self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.internal_alpha,
            self.ref_scores, self.sample_dist, self.sampled_data, self.weights, self.scores,
            self.ref_quantile, self.internal_quantile, self.selected_samples
        )
        with open(filename, 'wb') as h:
            pkl.dump(obj, h)

    def load(self, log_dir):
        filename = f'{log_dir}/{self.title}.cem'
        with open(filename, 'rb') as h:
            obj = pkl.load(h)

        self.title, self.original_dist, self.batch_size, self.w_clip, \
        self.ref_mode, self.ref_alpha, self.n_orig_per_batch, self.internal_alpha, \
        self.ref_scores, self.sample_dist, self.sampled_data, self.weights, self.scores, \
        self.ref_quantile, self.internal_quantile, self.selected_samples = obj

    def get_reference_q(self):
        if len(self.ref_quantile) > 0:
            return self.ref_quantile[-1]
        return -np.inf

    def get_internal_q(self):
        if len(self.internal_quantile) > 0:
            return self.internal_quantile[-1]
        return -np.inf

    def get_reference_alpha(self):
        return self.ref_alpha

    def get_sample_dist(self):
        raise NotImplementedError()

    ########   Sampling-related methods   ########
    def get_weight(self, x, use_original_dist=False):
        if use_original_dist:
            return np.ones(x.shape[0])

        lr = self.likelihood_ratio(x)
        if self.w_clip:
            lr = np.clip(lr, 1/self.w_clip, self.w_clip)
        return lr

    def likelihood_ratio(self, x):
        return self.pdf(x, self.original_dists[-1]) / \
               self.pdf(x, self.sample_dist[-1])

    def sample(self):
        '''Given dist. parameters, return a sample drawn from the distribution.'''
        raise NotImplementedError()

    def pdf(self, x, use_original_dist=False):
        '''Given a sample x and distribution parameters dist, return P(x|dist).'''
        raise NotImplementedError()

    ########   Update-related methods   ########
    def update_distribution(self, contexts, scores, aux_contexts, aux_scores):
        self.update(contexts, scores, aux_contexts, aux_scores)
        self.select_samples()

        samples = [self.sampled_data[-1][i, :] for i in range(self.sampled_data[-1].shape[0]) if self.selected_samples[-1][i]]
        weights = [self.weights[-1][i] for i in range(self.sampled_data[-1].shape[0]) if self.selected_samples[-1][i]]

        if len(samples) > 0:
            self.update_sample_distribution(np.array(samples), np.array(weights))
        else:
            self.sample_dist.append(self.sample_dist[-1])

        self.reset_batch()

    def update(self, contexts, scores, aux_contexts, aux_scores):
        self.sampled_data.append(np.concatenate((contexts, aux_contexts)))
        self.scores.append(np.concatenate((scores, aux_scores)))
        weights = self.get_weight(contexts, use_original_dist=True)
        aux_weights = self.get_weight(aux_contexts, use_original_dist=False)
        self.weights.append(np.concatenate((weights, aux_weights)))

    def reset_batch(self):
        self.sampled_data.append([])
        self.scores.append([])
        self.weights.append([])

    def select_samples(self):
        # Get internal quantile
        q_int = quantile(self.scores[-1], self.internal_alpha)

        # Get reference quantile from "external" data
        q_ref = -np.inf
        if self.ref_mode == 'train':
            q_ref = quantile(self.scores[-1][:self.n_orig_per_batch],
                             self.ref_alpha, estimate_underlying_quantile=True)
        elif self.ref_mode == 'valid':
            if self.ref_scores is None:
                warnings.warn('ref_mode=valid, but no '
                              'validation scores were provided.')
            else:
                q_ref = quantile(self.ref_scores, 100*self.ref_alpha,
                                 estimate_underlying_quantile=True)
        elif self.ref_mode == 'none':
            q_ref = -np.inf
        else:
            warnings.warn(f'Invalid ref_mode: {self.ref_mode}')

        # Take the max over the two
        self.internal_quantile.append(q_int)
        self.ref_quantile.append(q_ref)
        q = max(q_int, q_ref)

        # Select samples
        R = np.array(self.scores[-1])
        selection = R < q

        if self.force_min_samples:
            missing_samples = int(
                self.internal_alpha*self.batch_size - np.sum(selection))
            if missing_samples > 0:
                samples_to_add = np.where(R == q)[0]
                if missing_samples < len(samples_to_add):
                    samples_to_add = np.random.choice(
                        samples_to_add, missing_samples, replace=False)
                selection[samples_to_add] = True

        # Extension - Should this be done before forcing min samples or after?
        selection_target = self.select_target_contexts(self.sampled_data[-1])
        selection_final = selection*selection_target

        self.selected_samples.append(selection_final.ravel().tolist())

    # Extension
    def select_target_contexts(self, contexts):
        return np.exp(self.target_log_likelihood(contexts)) > 0.

    def update_ref_scores(self, scores):
        self.ref_scores = scores

    def update_sample_distribution(self, samples, weights):
        '''Return the parameters of a distribution given samples.'''
        raise NotImplementedError()

    def update_original_distribution(self, dist_params):
        raise NotImplementedError()

    def update_ref_alpha(self):
        raise NotImplementedError()

def quantile(x, q, w=None, is_sorted=False, estimate_underlying_quantile=False):
    n = len(x)
    # If we estimate_underlying_quantile, we refer to min(x),max(x) not as
    #  quantiles 0,1, but rather as quantiles 1/(n+1),n/(n+1) of the
    #  underlying distribution from which x is sampled.
    if estimate_underlying_quantile and n > 1:
        q = q * (n+1)/(n-1) - 1/(n-1)
        q = np.clip(q, 0, 1)
    # Unweighted quantiles
    if w is None:
        return np.percentile(x, 100*q)
    # Weighted quantiles
    x = np.array(x)
    w = np.array(w)
    if not is_sorted:
        ids = np.argsort(x)
        x = x[ids]
        w = w[ids]
    w = np.cumsum(w) - 0.5*w
    w -= w[0]
    w /= w[-1]
    return np.interp(q, w, x)
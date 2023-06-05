import torch
import numpy as np
from deep_sprl.util.torch import to_float_tensor
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution
from deep_sprl.util.cauchy_torch_distribution import CauchyTorchDistribution
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import os
import pickle
import time

from torchquad import MonteCarlo, set_up_backend


TORCH_DIST_DICT = {
    "gaussian": GaussianTorchDistribution,
    "cauchy": CauchyTorchDistribution,
}


def get_torch_dist(dist_type=None):
    if dist_type in TORCH_DIST_DICT:
        return TORCH_DIST_DICT[dist_type]
    else:
        raise ValueError(f"Given dist_type={dist_type} is not in {list(TORCH_DIST_DICT.keys())}.")


class FooResult:
    def __init__(self, fun, x, message) -> None:
        self.fun = fun
        self.x = x
        self.message = message


class AbstractSelfPacedTeacher:

    def __init__(self, init_mean, flat_init_chol, target_log_likelihood, target_sampler, max_kl, context_bounds,
                 callback=None, dist_type=None, ext_bounds=None):
        self.context_dist = get_torch_dist(dist_type)(init_mean, flat_init_chol,use_cuda=False, dtype=torch.float64)
        self.target_log_likelihood = target_log_likelihood
        self.target_sampler = target_sampler
        self.max_kl = max_kl
        self.callback = callback
        self.context_bounds = context_bounds
        self.ext_bounds = ext_bounds

    def target_context_kl(self, numpy=True):
        samples = self.context_dist.sample(sample_shape=(1000,))
        kl = torch.mean(self.context_dist.log_pdf_t(samples) -
                        torch.from_numpy(self.target_log_likelihood(samples.detach().numpy())))

        samples = self.target_sampler(1000)
        target_log_pdf = torch.from_numpy(self.target_log_likelihood(samples))
        cur_log_pdf = self.context_dist.log_pdf_t(torch.from_numpy(samples))
        kl2 = torch.mean(torch.exp(cur_log_pdf - target_log_pdf) * (cur_log_pdf - target_log_pdf))

        kl_t = 0.5 * kl + 0.5 * kl2

        # lb = self.context_bounds[0] - self.ext_bounds
        # ub = self.context_bounds[1] + self.ext_bounds
        #
        # def _kl_inner(x):
        #     return torch.exp(self.context_dist.log_pdf_t(x)) * (
        #             self.context_dist.log_pdf_t(x) - torch.from_numpy(self.target_log_likelihood(x)))
        # mc = MonteCarlo()
        # kl_t = mc.integrate(fn=_kl_inner, dim=lb.shape[0], integration_domain=[[lb[0], ub[0]], [lb[1], ub[1]]], N=5000)

        if numpy:
            return kl_t.detach().numpy()
        else:
            return kl_t

    def _compute_context_kl(self, old_context_dist):
        return torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, self.context_dist.distribution_t)

    def _compute_expected_performance(self, dist, cons_t, old_c_log_prob_t, c_val_t):
        con_ratio_t = torch.exp(dist.log_pdf_t(cons_t) - old_c_log_prob_t)
        return torch.mean(con_ratio_t * c_val_t)


class SelfPacedTeacherV2(AbstractTeacher, AbstractSelfPacedTeacher):

    def __init__(self, target_log_likelihood, target_sampler, initial_mean, initial_variance, context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, callback=None, dist_type="gaussian",
                 ext_bounds=None):

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_dim = initial_mean.shape[0]
        self.perf_lb = perf_lb
        self.perf_lb_reached = False
        self.callback = callback
        self.torch_dist = get_torch_dist(dist_type=dist_type)

        if std_lower_bound is not None and kl_threshold is None:
            raise RuntimeError("Error! Both Lower Bound on standard deviation and kl threshold need to be set")
        else:
            if std_lower_bound is not None:
                if isinstance(std_lower_bound, np.ndarray):
                    if std_lower_bound.shape[0] != self.context_dim:
                        raise RuntimeError("Error! Wrong dimension of the standard deviation lower bound")
                elif std_lower_bound is not None:
                    std_lower_bound = np.ones(self.context_dim) * std_lower_bound
            self.std_lower_bound = std_lower_bound
            self.kl_threshold = kl_threshold

        # Create the initial context distribution
        if isinstance(initial_variance, np.ndarray):
            flat_init_chol = self.torch_dist.flatten_matrix(initial_variance, tril=False)
        else:
            flat_init_chol = self.torch_dist.flatten_matrix(initial_variance * np.eye(self.context_dim), tril=False)

        super(SelfPacedTeacherV2, self).__init__(initial_mean, flat_init_chol, target_log_likelihood, target_sampler,
                                                 max_kl, context_bounds, callback=callback, dist_type=dist_type,
                                                 ext_bounds=ext_bounds)

    def old_kl_con(self, x, old_context_dist, obj=True, grad=False):
        dist = self.torch_dist.from_weights(self.context_dim, x, dtype=torch.float64)
        # kl_div = torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, dist.distribution_t)

        samples = old_context_dist.sample(sample_shape=(1000,))
        kl = torch.mean(old_context_dist.log_pdf_t(samples) - dist.log_pdf_t(samples))

        samples = dist.sample(sample_shape=(1000,))
        cur_log_pdf = dist.log_pdf_t(samples)
        old_log_pdf = old_context_dist.log_pdf_t(samples)
        kl2 = torch.mean(torch.exp(old_log_pdf - cur_log_pdf) * (old_log_pdf - cur_log_pdf))

        kl_div = 0.5 * kl + 0.5 * kl2

        # print(f"kl_div={kl_div} | old_context_dist={old_context_dist.parameters()} | dist={dist.parameters()}\n")
        if grad:
            mu_grad, chol_flat_grad = torch.autograd.grad(kl_div, dist.parameters())
            dx = np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()])
            if obj:
                return kl_div.detach().numpy(), dx
            else:
                return dx
        else:
            if obj:
                return kl_div.detach().numpy()
            else:
                raise RuntimeError("Either obj or grad need to be true!")

    def expected_performance(self, x, contexts_t, old_c_log_prob_t, c_val_t, obj=True, grad=False, negate=False):
        dist = self.torch_dist.from_weights(self.context_dim, x, dtype=torch.float64)
        perf = self._compute_expected_performance(dist, contexts_t, old_c_log_prob_t, c_val_t)
        if negate:
            perf = -1 * perf

        if grad:
            mu_grad, chol_flat_grad = torch.autograd.grad(perf, dist.parameters())
            dx = np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()])

            if obj:
                return perf.detach().numpy(), dx
            else:
                return dx
        else:
            if obj:
                return perf.detach().numpy()
            else:
                raise RuntimeError("Either obj or grad need to be true!")

    def update_distribution(self, contexts, values):
        old_context_dist = self.torch_dist.from_weights(self.context_dim, self.context_dist.get_weights(),
                                                        dtype=torch.float64)
        contexts_t = to_float_tensor(contexts, use_cuda=False, dtype=torch.float64)
        old_c_log_prob_t = old_context_dist.log_pdf_t(contexts_t).detach()

        # Estimate the value of the state after the policy update
        c_val_t = to_float_tensor(values, use_cuda=False, dtype=torch.float64)
        kl_constraint = NonlinearConstraint(lambda x: self.old_kl_con(x, old_context_dist), -np.inf, self.max_kl,
                                            jac=lambda x: self.old_kl_con(x, old_context_dist, obj=False, grad=True),
                                            keep_feasible=True)

        # Define the performance constraint
        perf_constraint = NonlinearConstraint(
            lambda x: self.expected_performance(x, contexts_t, old_c_log_prob_t, c_val_t), self.perf_lb, np.inf,
            jac=lambda x: self.expected_performance(x, contexts_t, old_c_log_prob_t, c_val_t, obj=False, grad=True),
            keep_feasible=False)

        if self.kl_threshold is not None and self.target_context_kl() > self.kl_threshold:
            # Define the variance constraint as bounds
            print("KL to target is greater than threshold.")
            cones = np.ones_like(self.context_dist.get_weights())
            lb = -np.inf * cones.copy()
            lb[self.context_dim: 2 * self.context_dim] = np.log(self.std_lower_bound)
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)

            # If the bounds are active, clip the standard deviation to be in bounds (because we may re-introduce
            # bounds after they have previously been removed)
            x0 = np.clip(self.context_dist.get_weights().copy(), lb, ub)
        else:
            x0 = self.context_dist.get_weights().copy()
            bounds = None

        try:
            if kl_constraint.fun(x0) >= self.max_kl:
                print("Warning! KL-Bound of x0 violates constraint already")

            if self.perf_lb_reached or perf_constraint.fun(x0) >= self.perf_lb:
                print("Optimizing KL")
                self.perf_lb_reached = True
                constraints = [kl_constraint, perf_constraint]

                # Define the objective plus Jacobian
                kl1_samples_t = old_context_dist.sample(sample_shape=(1000,))
                kl2_samples_t = torch.from_numpy(self.target_sampler(1000))
                kl1_log_pdf_t = old_context_dist.log_pdf_t(kl1_samples_t).detach()
                kl1_target_log_pdf_t = torch.from_numpy(self.target_log_likelihood(kl1_samples_t.detach().numpy()))
                kl2_target_log_pdf_t = torch.from_numpy(self.target_log_likelihood(kl2_samples_t.detach().numpy()))

                def objective(x):
                    dist = self.torch_dist.from_weights(self.context_dim, x, dtype=torch.float64)
                    kl1_new_log_pdf_t = dist.log_pdf_t(kl1_samples_t)
                    kl2_new_log_pdf_t = dist.log_pdf_t(kl2_samples_t)
                    kl1_t = torch.mean(
                        torch.exp(kl1_new_log_pdf_t - kl1_log_pdf_t) * (kl1_new_log_pdf_t - kl1_target_log_pdf_t))
                    kl_target_t = torch.mean(
                        torch.exp(kl2_new_log_pdf_t - kl2_target_log_pdf_t) * (
                                kl2_new_log_pdf_t - kl2_target_log_pdf_t))
                    kl_t = 0.5 * (kl1_t + kl_target_t)

                    mu_grad, chol_flat_grad = torch.autograd.grad(kl_t, dist.parameters())
                    return kl_t.detach().numpy() + 0.5, \
                           np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()]).astype(
                               np.float64)

                # def objective(x):
                #     lb = self.context_bounds[0] - self.ext_bounds
                #     ub = self.context_bounds[1] + self.ext_bounds
                #     dist = self.torch_dist.from_weights(self.context_dim, x, dtype=torch.float64)
                #
                #     def _kl_inner(x):
                #         return torch.exp(dist.log_pdf_t(x)) * (
                #                 dist.log_pdf_t(x) - torch.from_numpy(self.target_log_likelihood(x)))
                #
                #     mc = MonteCarlo()
                #     kl_t = mc.integrate(fn=_kl_inner, dim=lb.shape[0],
                #                         integration_domain=[[lb[0], ub[0]], [lb[1], ub[1]]], N=5000)
                #     mu_grad, chol_flat_grad = torch.autograd.grad(kl_t, dist.parameters())
                #     return kl_t.detach().numpy(), \
                #            np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()]).astype(
                #                np.float64)

                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": 1e-4, "xtol": 1e-6})
            # Only do the optimization of the context distribution if the performance threshold has not yet been reached
            # even once
            else:
                print("Optimizing performance")
                # Define the objective plus Jacobian
                objective = lambda x: self.expected_performance(x, contexts_t, old_c_log_prob_t, c_val_t, grad=True,
                                                                negate=True)
                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=[kl_constraint], options={"gtol": 1e-4, "xtol": 1e-6})
        except Exception as e:
            os.makedirs("opt_errors", exist_ok=True)
            with open(os.path.join("opt_errors", "error_" + str(time.time())), "wb") as f:
                pickle.dump((self.context_dist.get_weights(), contexts, values), f)
            print("Exception occurred during optimization! Storing state and re-raising!")
            raise e
            # old_dist_weight = self.context_dist.get_weights().copy()
            # res = FooResult(fun=objective(old_dist_weight)[0], x=old_dist_weight, message="Exception occurred during optimization!")

        # If it was not a success, but the objective value was improved and the bounds are still valid, we still
        # use the result
        old_f = objective(self.context_dist.get_weights())[0]
        std_ok = bounds is None or (np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub))
        if self.perf_lb_reached:
            old_perf = perf_constraint.fun(old_context_dist.get_weights())
            new_perf = perf_constraint.fun(res.x)
            if old_perf < 0.95 * self.perf_lb and new_perf < 0.95 * self.perf_lb:
                print("Could not recover from performance loss in optimization. Will keep old distribution")
                perf_ok = False
            else:
                perf_ok = True
        else:
            perf_ok = True

        if perf_ok and std_ok and res.fun <= old_f:
            self.context_dist.set_weights(res.x)
        else:
            print(f"Warning! Context optimization unsuccessful - will keep old values. Message: {res.message} | "
                  f"perf_ok={perf_ok} | std_ok={std_ok} | (res.fun <= old_f)={res.fun <= old_f}")

        if self.callback is not None:
            self.callback(old_context_dist.mean(), old_context_dist.covariance_matrix(),
                          self.context_dist.mean(), self.context_dist.covariance_matrix(), contexts, values)

    def sample(self):
        sample_ok = False
        count = 0
        while not sample_ok and count < 100:
            sample = self.context_dist.sample().detach().numpy()
            sample_ok = np.all(self.context_bounds[0] <= sample) and (np.all(sample <= self.context_bounds[1]))
            count += 1

        if sample_ok:
            return sample
        else:
            mu = self.context_dist.mean()
            # Why uniform sampling? Because if we sample 100 times outside of the allowed
            if np.all(self.context_bounds[0] <= mu) and (np.all(mu <= self.context_bounds[1])):
                return np.random.uniform(self.context_bounds[0], self.context_bounds[1])
            else:
                return np.clip(sample, self.context_bounds[0], self.context_bounds[1])

    def save(self, path):
        weights = self.context_dist.get_weights()
        np.save(os.path.join(path, "teacher"), weights)

    def load(self, path):
        self.context_dist.set_weights(np.load(os.path.join(path, "teacher.npy")))

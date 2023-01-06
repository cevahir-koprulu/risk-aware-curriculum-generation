from abc import ABC
import numpy as np
import torch
from deep_sprl.util.torch import to_float_tensor

class AbstractDistribution(object):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization
    algorithms in order to perform exploration in parameter space. In
    literature, they are also known as high level policies.

    """

    def sample(self):
        """
        Draw a sample from the distribution.

        Returns:
            A random vector sampled from the distribution.

        """
        raise NotImplementedError

    def log_pdf(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            x (np.ndarray): the point where the log pdf is calculated

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Compute the probability density function in the specified point

        Args:
            x (np.ndarray): the point where the pdf is calculated

        Returns:
            The value of the pdf in the specified point.

        """
        raise np.exp(self.log_pdf(x))


class Distribution(AbstractDistribution):
    """
    Interface for Distributions to represent a generic probability distribution.
    Probability distributions are often used by black box optimization
    algorithms in order to perform exploration in parameter space. In
    literature, they are also known as high level policies.

    """

    def mle(self, theta, weights=None):
        """
        Compute the (weighted) maximum likelihood estimate of the points,
        and update the distribution accordingly.

        Args:
            theta (np.ndarray): a set of points, every row is a sample
            weights (np.ndarray, None): a vector of weights. If specified
                                        the weighted maximum likelihood
                                        estimate is computed instead of the
                                        plain maximum likelihood. The number of
                                        elements of this vector must be equal
                                        to the number of rows of the theta
                                        matrix.

        """
        raise NotImplementedError

    def diff_log(self, theta):
        """
        Compute the derivative of the gradient of the probability denstity
        function in the specified point.

        Args:
            theta (np.ndarray): the point where the gradient of the log pdf is calculated

        Returns:
            The gradient of the log pdf in the specified point.

        """
        raise NotImplementedError

    def diff(self, theta):
        """
        Compute the derivative of the probability density function, in the
        specified point. Normally it is computed w.r.t. the
        derivative of the logarithm of the probability density function,
        exploiting the likelihood ratio trick, i.e.:

        .. math::
            \\nabla_{\\rho}p(\\theta)=p(\\theta)\\nabla_{\\rho}\\log p(\\theta)

        Args:
            theta (np.ndarray): the point where the gradient of the pdf is
            calculated.

        Returns:
            The gradient of the pdf in the specified point.

        """
        return self(theta) * self.diff_log(theta)

    def get_parameters(self):
        """
        Getter.

        Returns:
             The current distribution parameters.

        """
        raise NotImplementedError

    def set_parameters(self, rho):
        """
        Setter.

        Args:
            rho (np.ndarray): the vector of the new parameters to be used by
                              the distribution

        """
        raise NotImplementedError

    @property
    def parameters_size(self):
        """
        Property.

        Returns:
             The size of the distribution parameters.

        """
        raise NotImplementedError


class TorchDistribution(AbstractDistribution, ABC):
    """
    Interface for a generic PyTorch distribution.
    A PyTorch distribution is a distribution implemented using PyTorch.
    Functions ending with '_t' use tensors as input, and also as output when
    required.

    """

    def __init__(self, use_cuda):
        """
        Constructor.

        Args:
            use_cuda (bool): whether to use cuda or not.

        """
        self._use_cuda = use_cuda

    def entropy(self):
        """
        Compute the entropy of the policy.

        Returns:
            The value of the entropy of the policy.

        """

        return self.entropy_t().detach().cpu().numpy()

    def entropy_t(self):
        """
        Compute the entropy of the policy.

        Returns:
            The tensor value of the entropy of the policy.

        """
        raise NotImplementedError

    def mean(self):
        """
        Compute the mean of the policy.

        Returns:
            The value of the mean of the policy.

        """
        return self.mean_t().detach().cpu().numpy()

    def mean_t(self):
        """
        Compute the mean of the policy.

        Returns:
            The tensor value of the mean of the policy.

        """
        raise NotImplementedError

    def log_pdf(self, x):
        x = to_float_tensor(x, self._use_cuda)
        return self.log_pdf_t(x).detach().cpu().numpy()

    def log_pdf_t(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point

        Args:
            x (torch.Tensor): the point where the log pdf is calculated

        Returns:
            The value of the log pdf in the specified point.

        """
        raise NotImplementedError

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the distribution

        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the trainable distribution parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._use_cuda

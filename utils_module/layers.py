import torch
import torch.nn.functional as F
from torch import nn


class LayerBnn(nn.Module):
    def __init__(self, bias, mu_prior, sigma_prior):
        super(LayerBnn, self).__init__()
        self.bias = bias
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.kl = None

    def sample(self, mu, rho):
        """
        [args]
            - mu(float): mean of the distribution
            - rho(float): rho of the distribution
        [objective]
        This method returns a sample from the gaussian distribution(mu, rho)
        """
        eps = torch.normal(0., 1., size=mu.size()).type_as(mu)
        value = mu + self.rho_to_std(rho) * eps
        return value

    @staticmethod
    def init_parameter(init_value, init_type, size):
        """
        [args]
            - init_value(float): initialization value
            - init_type([fixed, normal]): initialization strategy
            - size(tupe): tuple representing the size of the layer weights
        [objective]
        This method initializes the weights with a given strategy
        """
        if init_type == 'fixed':
            return nn.Parameter(torch.ones(size) * init_value)
        elif init_type == 'normal':
            return nn.Parameter(torch.normal(init_value, 0.01, size=size))
        else:
            raise ValueError('To implement')

    @staticmethod
    def norm_log_prob(t, m, s):
        if not(torch.is_tensor(s)):
            s = torch.tensor(s)
        log_prob = - 0.5 * torch.log(torch.pi * 2 * s**2) - (t - m)**2 / (2 * s**2)
        return log_prob.sum()

    @staticmethod
    def rho_to_std(rho):
        """
        [args]
            - rho(float): rho matrix
        [objective]
        This method converts the rho matrix to the standard deviation matrix (sigma) using the
        reparametrization function
        """
        return torch.where(rho < 50, torch.log1p(torch.exp(rho)), rho)


class LinearBnn(LayerBnn):
    def __init__(self,
                 in_size,
                 out_size,
                 init_rho_post,
                 init_mu_post,
                 sigma_prior,
                 mu_prior,
                 init_type='fixed',
                 bias=False):
        """
        [args]:
            - in_size(integer): input size of the layer
            - out_size(integer): output size of the layer
            - init_rho_post(float): initialization rho for the variational posterior
            - init_mu_post(float): initialization mu for the variational posterior
            - sigma_prior(float): standard deviation of the prior
            - mu_prior(float): mean of the prior
            - N(integer): number of neurons
            - p(integer): number of data points
            - alpha(float):
            - init_type([fixed, normal]): strategy for the weights initialization
            - bias(Boolean): add bias to the layer
        """
        super(LinearBnn, self).__init__(bias, mu_prior, sigma_prior)

        self.weight_mu = LinearBnn.init_parameter(init_mu_post, init_type, (out_size, in_size))
        self.weight_rho = LinearBnn.init_parameter(init_rho_post, init_type, (out_size, in_size))

        if bias:
            self.bias_mu = LinearBnn.init_parameter(init_mu_post, init_type, (out_size,))
            self.bias_rho = LinearBnn.init_parameter(init_mu_post, init_type, (out_size,))

    def forward(self, x):
        """
        [args]
        [objective]
        """
        w = self.sample(self.weight_mu, self.weight_rho)
        log_var_post = LinearBnn.norm_log_prob(w, self.weight_mu, LinearBnn.rho_to_std(self.weight_rho))
        log_prior = LinearBnn.norm_log_prob(w, self.mu_prior, self.sigma_prior)
        self.kl = log_var_post - log_prior

        if self.bias:
            b = self.sample(self.bias_mu, self.bias_rho)
            log_var_post = LinearBnn.norm_log_prob(b, self.bias_mu, LinearBnn.rho_to_std(self.bias_rho))
            log_prior = LinearBnn.norm_log_prob(b, self.mu_prior, self.sigma_prior)
            self.kl += log_var_post - log_prior
            out = F.linear(x, w, b)
        else:
            out = F.linear(x, w, None)
        return out


class ConvBnn(LayerBnn):
    def __init__(self,
                 in_size,
                 out_size,
                 init_rho_post,
                 init_mu_post,
                 sigma_prior,
                 mu_prior,
                 stride=1,
                 padding=0,
                 dilation=1,
                 kernel_size=3,
                 init_type='fixed',
                 bias=False):
        """
        [args]:
            - in_size(integer): input size of the layer
            - out_size(integer): output size of the layer
            - init_rho_post(float): initialization rho for the variational posterior
            - init_mu_post(float): initialization mu for the variational posterior
            - sigma_prior(float): standard deviation of the prior
            - mu_prior(float): mean of the prior
            - N(integer): number of neurons
            - p(integer): number of data points
            - alpha(float):
            - init_type([fixed, normal]): strategy for the weights initialization
        """
        super(ConvBnn, self).__init__(bias, mu_prior, sigma_prior)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.weight_mu = self.init_parameter(init_mu_post, init_type, (out_size, in_size, kernel_size, kernel_size))
        self.weight_rho = self.init_parameter(init_rho_post, init_type, (out_size, in_size, kernel_size, kernel_size))

        if bias:
            self.bias_mu = self.init_parameter(init_mu_post, init_type, (out_size,))
            self.bias_rho = self.init_parameter(init_mu_post, init_type, (out_size,))

    def forward(self, x):
        """
        [args]
        [objective]
        """
        w = self.sample(self.weight_mu, self.weight_rho)
        log_var_post = ConvBnn.norm_log_prob(w, self.weight_mu, ConvBnn.rho_to_std(self.weight_rho))
        log_prior = ConvBnn.norm_log_prob(w, self.mu_prior, self.sigma_prior)
        self.kl = log_var_post - log_prior

        if self.bias:
            b = self.sample(self.bias_mu, self.bias_rho)
            log_var_post = ConvBnn.norm_log_prob(b, self.bias_mu, ConvBnn.rho_to_std(self.bias_rho))
            log_prior = ConvBnn.norm_log_prob(b, self.mu_prior, self.sigma_prior)
            self.kl += log_var_post - log_prior
            out = F.conv2d(x, w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            out = F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return out

from torch.optim import Adam
from torch import nn
from truncated_normal import TruncatedNormal
from torch.autograd import Variable
import torch
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import numpy as np


class UniformSampler:

    def __init__(self, funcs: list, var_dim: int, device):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim

    def sample_var(self, batch_size: int):
        with torch.no_grad():
            vars = []
            for f in self.funcs:
                vars.append(
                    f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(self.var_dim)]))
        return [[f.to(self.device).requires_grad_() for f in fs] for fs in vars]

    def update(self, loss):
        pass


class VariableUniformSampler:

    def __init__(self, funcs: list, var_dim: int, device):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.mu = 0

    def sample_var(self, batch_size: int):
        with torch.no_grad():
            vars = []
            for f in self.funcs:
                mu = self.mu * torch.ones((batch_size, self.var_dim)).to(self.device)
                dist = TruncatedNormal(mu, 1, 0, 1)
                self.sample = dist.sample()
                vars.append(
                    f([self.sample[:, i].detach().unsqueeze(1) for i in range(self.var_dim)]))
        return [[f.requires_grad_() for f in fs] for fs in vars]

    def update(self, loss):
        indices = torch.argmax(loss)
        var_biases = self.sample[indices].expand_as(self.sample).detach()
        self.mu = var_biases

SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "variable": VariableUniformSampler
}

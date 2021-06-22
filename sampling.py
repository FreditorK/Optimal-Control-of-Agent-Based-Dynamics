from torch.optim import Adam
from torch import nn
from truncated_normal import TruncatedNormal
import torch
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
        vars = []
        for f in self.funcs:
            vars.append(
                f([torch.rand(size=(batch_size, 1)).to(self.device).requires_grad_() for _ in range(self.var_dim)]))
        return vars

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
        self.net = nn.Linear(var_dim, var_dim)
        self.optimiser = Adam(self.net.parameters(), lr=0.1)
        self.sample = None
        self.sigma_net = nn.Sequential(
            nn.Linear(8*var_dim, 16*var_dim),
            nn.ELU(),
            nn.Linear(16*var_dim, var_dim),
            nn.Sigmoid()
        )  # standard deviation
        self.mu_net = nn.Sequential(
            nn.Linear(8*var_dim, 16*var_dim),
            nn.ELU(),
            nn.Linear(16*var_dim, var_dim),
            nn.Sigmoid())  # gaussian mean
        self.dist = None

    def sample_var(self, batch_size: int):
        vars = []
        for f in self.funcs:
            sample = torch.rand(size=(batch_size, 8*self.var_dim)).to(self.device)
            mu = self.mu_net(sample)
            sigma = self.sigma_net(sample)
            self.dist = TruncatedNormal(mu, sigma, 0, 1)
            self.sample = self.dist.sample()
            vars.append(
                f([self.sample[:, i].unsqueeze(1).requires_grad_() for i in range(self.var_dim)]))
        return vars

    def update(self, loss):
        indices = torch.argsort(loss, descending=True)[:16]
        var_biases = self.sample[indices].detach()
        loss = -self.dist.log_prob(var_biases).mean()
        # maximise the liklihood of sampling from these points
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "variable": VariableUniformSampler
}

from torch.optim import Adam
from torch import nn
import torch


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

    def sample_var(self, batch_size: int):
        vars = []
        for f in self.funcs:
            vars.append(
                f([torch.rand(size=(batch_size, 1)).to(self.device).requires_grad_() for _ in range(self.var_dim)]))
        return vars


class GenerativeSampler:

    def __init__(self, funcs: list, var_dim: int, device):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ELU(),
            nn.Linear(16, var_dim),
            nn.Sigmoid()
        )

        self.optimizer = Adam(self.net.parameters(), lr=0.01)

    def sample_var(self, batch_size: int):
        vars = []
        with torch.no_grad():
            noise = torch.rand(size=(batch_size, 8))
            sample = self.net(noise).to(self.device)
        for f in self.funcs:
            vars.append(f([sample[:, i].unsqueeze(1).requires_grad_() for i in range(self.var_dim)]))
        return vars

    def sample_update(self, batch_size: int):
        vars = []
        noise = torch.rand(size=(batch_size, 8)).requires_grad_()
        sample = self.net(noise).to(self.device)
        for f in self.funcs:
            vars.append(f([sample[:, i].unsqueeze(1) for i in range(self.var_dim)]))
        return vars

    def update(self, loss):
        self.optimizer.zero_grad()
        (-loss).backward()
        self.optimizer.step()


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "generative": GenerativeSampler
}

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Function


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


class GRLLayer(nn.Module):
    def __init__(self):
        super(GRLLayer, self).__init__()

    def forward(self, x):
        return RevGrad.apply(x)


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
            vars.append(f([torch.rand(size=(batch_size, 1)).to(self.device).requires_grad_() for _ in range(self.var_dim)]))
        return vars


class GenerativeSampler(nn.Module):
    def __init__(self, output_dim, domain_function, device):
        """
        Sample Generator
        int:param output_dim: dimension of vector required by domain
        lambda:param domain_function: function that maps [-1, 1] to the domain/boundary
        """
        super().__init__()
        self.domain_function = domain_function
        self.device = device
        self.input_dim = 8
        self.net = nn.Sequential(
            GRLLayer(),
            nn.Linear(self.input_dim, 16),
            nn.ELU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )
        self.sampler_optimizer = Adam(self.parameters(), lr=0.1)

    def sample_x(self, batch_size):
        x = torch.rand(size=(batch_size, self.input_dim)).to(self.device).requires_grad_()
        x = self.net(x)
        return self.domain_function(x)

    def sample_t(self, batch_size):
        return torch.rand(size=(batch_size, 1)).to(self.device).requires_grad_()

    def update(self, domain_loss):
        self.sampler_optimizer.zero_grad()
        domain_loss.backward(retain_graph=True)
        self.sampler_optimizer.step()


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "generative": GenerativeSampler
}

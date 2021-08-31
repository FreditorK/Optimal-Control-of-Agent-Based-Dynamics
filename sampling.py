import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.quasirandom import SobolEngine

PATH_SPACES = {
    "AC": {
        "SDE": lambda X, u, t, dt, dW: 0 * X * dt + torch.einsum("bij, bj -> bi", torch.diag_embed(X * 0 + 1), dW),
        "terminal_time": 0.3,
        "N_range": (15, 30),
        "control": lambda J, X, t: 0,
        "domain": (-np.inf, np.inf),
        "weight": 1
    }
}


class UniformSampler:

    def __init__(self, funcs: list, var_dim: int, device, *args):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for f, batch_size in self.funcs:
                func_list.append(
                    f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(self.var_dim)]))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def update(self, *args):
        pass


class QuasiUniformSampler:

    def __init__(self, funcs: list, var_dim: int, device, *args):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.sobol = SobolEngine(1, scramble=True)

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for f, batch_size in self.funcs:
                func_list.append(
                    f([self.sobol.draw(batch_size) for _ in range(self.var_dim)]))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def update(self, *args):
        pass


class GaussianSampler:

    def __init__(self, funcs: list, var_dim: int, device, *args):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for f, batch_size in self.funcs:
                func_list.append(
                    f([torch.zeros(size=(batch_size, 1)).normal_(mean=0.0, std=1.0) for _ in
                       range(self.var_dim - 1)] + [torch.zeros(size=(batch_size, 1)).uniform_()]))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def update(self, *args):
        pass


class PathSampler:
    def __init__(self, funcs: list, var_dim: int, device, *args):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.name = args[0]
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.sde = PATH_SPACES[self.name]["SDE"]
        self.current_batch = [f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(var_dim)]) for f, batch_size in self.funcs]
        self.alpha = [torch.tensor(1.0, requires_grad=True) for _, batch_size in self.funcs]
        self.alpha_optimizer = Adam(self.alpha, lr=PATH_SPACES[self.name]["lr"])
        self.terminal_time = PATH_SPACES[self.name]["terminal_time"]
        self.N_range = PATH_SPACES[self.name]["N_range"]
        self.opt_control = PATH_SPACES[self.name]["control"]
        self.domain = PATH_SPACES[self.name]["domain"]
        self.us = [torch.zeros(size=(batch_size, 1)) for f, batch_size in self.funcs]
        self.boundary_sampler_dependency = False
        self.boundary_memory = torch.zeros(size=(1, var_dim))
        #self.app = [] # this can be removed, plotting purposes
        #self.app_2 = [] # this can be removed, plotting purposes

    def sample_var(self):
        self.alpha_optimizer.zero_grad()
        with torch.no_grad():
            func_list = []
            for idx, ((f, batch_size), u) in enumerate(zip(self.funcs, self.us)):
                func_list.append(self.sample_batch(batch_size, u, f, idx))
        self.alpha_optimizer.step()
        return [[v.to(self.device).detach().requires_grad_() for v in fs] for fs in func_list]

    def sample_batch(self, batch_size, u, f, idx):
        dt = self.terminal_time / (self.N_range*torch.ones((batch_size, 1)))
        sqrt_dt = torch.sqrt(dt)
        dW = sqrt_dt * torch.randn(batch_size, self.var_dim - 1)
        X_old = torch.cat(self.current_batch[idx][:-1], dim=-1)
        with torch.enable_grad():
            X = X_old + self.sde(X_old, (1-self.alpha[idx])*u, self.current_batch[idx][-1], dt, dW)
        new_mask = (self.current_batch[idx][-1] + dt >= self.terminal_time).long()
        old_mask = -(new_mask - 1)
        new_X = torch.cat(f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(self.var_dim)])[:-1], dim=-1)
        X_masked = old_mask * X + new_mask * new_X

        with torch.enable_grad():
            alpha_loss = (self.alpha[idx]**2).mean() #+ self.weight*self.weigthed_p_ws_dist(X, self.current_batch[idx][-1]).mean()
            alpha_loss.backward()

        self.current_batch[idx] = [torch.clamp(X_masked[:, None, i], min=self.domain[0], max=self.domain[1]) for i in
                                   range(self.var_dim - 1)] + [torch.fmod(self.current_batch[idx][-1]+ dt, self.terminal_time)]

        if self.boundary_sampler_dependency:
            terminal = X_masked[new_mask.flatten() == 1, :]
            length = terminal.shape[0]
            if length > 0:
                boundary_batch = torch.cat((terminal, torch.ones(length, 1) * self.terminal_time), dim=-1)
                self.boundary_memory = torch.cat((boundary_batch, self.boundary_memory[:-length]), dim=0)

        #self.app.append(torch.clamp(X[0], min=-1, max=1)) # this can be removed

        return self.current_batch[idx]

    def update(self, Js, var_samples, i):
        self.us = [self.opt_control(J, v[:-1], v[-1]).detach().cpu() for J, v in zip(Js, var_samples)]
        #if i == 100 or i == 500 or i == 1400:
            #self.app_2.append(torch.cat(self.current_batch[0], dim=-1))  # this can be removed

    def weigthed_p_ws_dist(self, X, ts, k=6, p=2):
        ts, ts_idxs = torch.topk(ts, k, dim=0)
        X_reordered = X[ts_idxs.flatten()]
        X_reordered, _ = torch.sort(X_reordered, dim=-1)
        X_conv = X_reordered[1:] - X_reordered[:-1]
        dist = (ts[1:]/self.terminal_time)*torch.mean(torch.abs(X_conv) ** p, dim=-1, keepdims=True)  # **(1/p)
        return dist


class TerminalPathSampler:
    def __init__(self, funcs: list, var_dim: int, device, *args):
        """
        :param funcs: sampling functions for subdomains
        :param var_funcs: sampling functions for each variable
        :param device: cpu/gpu
        """
        self.name = args[0]
        self.domain_sampler = args[1]
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.current_batch = [f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(var_dim)]) for f, batch_size in self.funcs]
        self.domain_sampler.boundary_sampler_dependency = True
        self.domain_sampler.boundary_memory = torch.cat([torch.cat(f([torch.zeros(size=(4*batch_size, 1)).uniform_() for _ in range(var_dim)]), dim=-1) for f, batch_size in self.funcs], dim=0)

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for idx, (f, batch_size) in enumerate(self.funcs):
                func_list.append(self.sample_batch(batch_size, f, idx))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def sample_batch(self, batch_size, f, idx):
        size = self.domain_sampler.boundary_memory.shape[0]
        indices = torch.randint(0, size, (batch_size, ))
        X = self.domain_sampler.boundary_memory[indices]
        self.current_batch[idx] = [X[:, None, i] for i in range(self.var_dim)]
        return self.current_batch[idx]


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "quasiuniform": QuasiUniformSampler,
    "path": PathSampler,
    "gaussian": GaussianSampler,
    "uniform_bound": UniformSampler,
    "quasiuniform_bound": QuasiUniformSampler,
    "path_bound": TerminalPathSampler,
    "gaussian_bound": GaussianSampler,
}

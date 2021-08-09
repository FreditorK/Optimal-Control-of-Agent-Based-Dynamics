import torch
import numpy as np

PATH_SPACES = {
    "AC": {
        "SDE": lambda X, u, t, dt, dW: 0 * X * dt + torch.einsum("bij, bj -> bi", torch.diag_embed(X * 0 + 1), dW),
        "terminal_time": 0.3,
        "N_range": (15, 30),
        "control": lambda J, X, t: 0,
        "domain": (-np.inf, np.inf)
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
        self.bound = args[1]
        self.device = device
        self.funcs = funcs
        self.var_dim = var_dim
        self.sde = PATH_SPACES[self.name]["SDE"]
        self.current_batch = [f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(var_dim)]) for f, batch_size in self.funcs]
        self.terminal_time = PATH_SPACES[self.name]["terminal_time"]
        self.N_range = PATH_SPACES[self.name]["N_range"]
        self.opt_control = PATH_SPACES[self.name]["control"]
        self.domain = PATH_SPACES[self.name]["domain"]
        self.us = [torch.zeros(size=(batch_size, 1)) for f, batch_size in self.funcs]
        self.alpha = 1

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for idx, ((f, batch_size), u) in enumerate(zip(self.funcs, self.us)):
                func_list.append(self.sample_batch(batch_size, u, f, idx))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def sample_batch(self, batch_size, u, f, idx):
        dt = self.terminal_time / torch.randint(low=self.N_range[0], high=self.N_range[1], size=(batch_size, 1))
        sqrt_dt = torch.sqrt(dt)
        dW = sqrt_dt * torch.randn(batch_size, self.var_dim - 1)
        X = torch.cat(self.current_batch[idx][:-1], dim=-1)
        X = X + self.sde(X, u, self.current_batch[idx][-1], dt, dW)
        new_mask = torch.where(self.current_batch[idx][-1] > self.terminal_time, 1, 0)
        old_mask = -(new_mask - 1)
        new_X = torch.cat(f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(self.var_dim-1)]), dim=-1)
        X = old_mask * X + new_mask * new_X
        self.current_batch[idx] = [torch.clamp(X[:, None, i], min=self.domain[0], max=self.domain[1]) for i in range(self.var_dim - 1)] + [torch.fmod(self.current_batch[idx][-1] + dt, self.terminal_time)]

        return self.current_batch[idx]

    def update(self, Js, var_samples, i):
        #print([torch.mean(self.opt_control(J, v[:-1], v[-1]), dim=0).detach().cpu() for J, v in zip(Js, var_samples)])
        #print(self.alpha)  + (torch.rand_like(v[0].detach().cpu())-0.5)*(1/15)*loss
        self.us = [(1-self.alpha)*self.opt_control(J, v[:-1], v[-1]).detach().cpu() for J, v in zip(Js, var_samples)] #[-0.1*(torch.cat(v[:-1], dim=-1) - 0.2).detach().cpu() for J, v in zip(Js, var_samples)] #
        self.alpha *= 0.99
        #self.alpha = (self.alpha*1.02) if self.alpha < 0.9 else 1.0


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "path": PathSampler,
    "gaussian": GaussianSampler
}

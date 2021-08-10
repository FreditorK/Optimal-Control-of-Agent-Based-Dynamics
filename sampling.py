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
        self.boundary_memory = torch.zeros(size=(1, var_dim))
        self.app = []
        print(self.current_batch[0][-1][0])

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
        new_mask = (self.current_batch[idx][-1] + dt >= self.terminal_time).long() # (sum([(v-0.2)**2 for v in self.current_batch[idx][:-1]]) <= 0.5).long()
        old_mask = -(new_mask - 1)
        new_X = torch.cat(f([torch.zeros(size=(batch_size, 1)).uniform_() for _ in range(self.var_dim)])[:-1], dim=-1)
        X = old_mask * X + new_mask * new_X
        self.current_batch[idx] = [torch.clamp(X[:, None, i], min=self.domain[0], max=self.domain[1]) for i in
                                   range(self.var_dim - 1)] + [old_mask*(self.current_batch[idx][-1]+dt)]
        '''
        terminal = X[new_mask.flatten() == 1, :]
        length = terminal.shape[0]
        if length > 0:
            boundary_batch = torch.cat((terminal, torch.ones(length, 1)*self.terminal_time), dim=-1)
            self.boundary_memory = torch.cat((boundary_batch, self.boundary_memory[:-length]), dim=0)
        else:
            return self.sample_batch(batch_size, u, f, idx)
        '''
        self.app.append(torch.clamp(X[0], min=-1, max=1))
        return self.current_batch[idx]

    def update(self, Js, var_samples, i):
        self.us = [(1-self.alpha)*self.opt_control(J, v[:-1], v[-1]).detach().cpu() for J, v in zip(Js, var_samples)] #[-0.1*(torch.cat(v[:-1], dim=-1) - 0.2).detach().cpu() for J, v in zip(Js, var_samples)] #[-6*(torch.cat(v[:-1], dim=-1) - 0.2).detach().cpu() for J, v in zip(Js, var_samples)] #
        self.alpha *= 0.999
        #if i % 10 == 0:
            #print([ self.current_batch[0][i][0] for i in range(self.var_dim-1)])


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
        for _ in range(10):
            self.domain_sampler.sample_var()
        return self.current_batch[idx]


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "path": PathSampler,
    "gaussian": GaussianSampler,
    "uniform_bound": UniformSampler,
    "path_bound": TerminalPathSampler,
    "gaussian_bound": GaussianSampler,
}

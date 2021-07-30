import torch
import os
import pandas as pd
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PATH_SPACES = {
    "AC": {
        "drift": lambda X, u, t: 0 * X,
        "diffusion": lambda X, u, t: torch.diag_embed(X * 0 + 1),
        "init_sample": lambda batch_size, var_dim: 0*torch.rand(batch_size, var_dim),
        "terminal_time": 0.3,
        "noise": 0.0
    },
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
        path_str = os.path.join(os.getcwd(), "{}_{}.csv".format(str(self.bound) + self.name, var_dim))
        path = Path(path_str)
        if not path.is_file():
            num_samples = 60 if self.bound else 20
            self.create_dataset(path_str, self.name, num_samples)
        self.dataset = pd.read_csv(path_str)
        self.idxs = np.array(range(self.dataset.shape[0]))
        np.random.shuffle(self.idxs)

    def sample_var(self):
        with torch.no_grad():
            func_list = []
            for f, batch_size in self.funcs:
                func_list.append(f(self.sample_batch(batch_size)))
        return [[v.to(self.device).requires_grad_() for v in fs] for fs in func_list]

    def sample_batch(self, batch_size):
        if len(self.idxs) < batch_size:
            self.idxs = np.array(range(self.dataset.shape[0]))
            np.random.shuffle(self.idxs)
        batch_idxs = self.idxs[:batch_size]
        self.idxs = self.idxs[batch_size:]
        data = self.dataset.iloc[batch_idxs, 1:].to_numpy()
        return [torch.from_numpy(data[:, None, i]).float() for i in range(data.shape[1])]

    def create_dataset(self, path, name, num_samples):
        b = PATH_SPACES[name]["drift"]
        sigma = PATH_SPACES[name]["diffusion"]
        data = []
        d_steps = np.random.randint(int(PATH_SPACES[name]["terminal_time"]*50), int(PATH_SPACES[name]["terminal_time"]*100), num_samples)
        for sample in range(num_samples):
            X_sample = torch.zeros((d_steps[sample], 128, self.var_dim - 1))
            X_sample[0, :, :] = PATH_SPACES[name]["init_sample"](128, self.var_dim-1)
            ts = torch.zeros((d_steps[sample], 128, 1))
            dt = PATH_SPACES[name]["terminal_time"] / d_steps[sample]
            sqrt_dt = np.sqrt(dt)
            X = X_sample[0]
            for i in range(1, d_steps[sample]):
                t = dt * i
                dW = sqrt_dt * torch.randn(128, self.var_dim - 1)
                u = PATH_SPACES[name]["noise"]*torch.rand(128, self.var_dim - 1)
                X = X + b(X, u, t) * dt + torch.einsum("bij, bj -> bi", sigma(X, u, t), dW)
                X_sample[i] = X
                ts[i] = t
            Xt = torch.cat((X_sample, ts), dim=-1)
            XT = Xt[-1]
            if self.bound:
                data.append(XT)
            else:
                data.append(Xt)

        pd.DataFrame(torch.cat(data, dim=0).numpy().reshape((-1, self.var_dim))).to_csv(path)


SAMPLING_METHODS = {
    "uniform": UniformSampler,
    "path": PathSampler
}

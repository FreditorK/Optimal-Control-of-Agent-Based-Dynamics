import torch


class Sampler:

    def __init__(self, x_dim):
        self.x_dim = x_dim

    def sample_t(self, batch_size):
        return torch.rand(size=(batch_size, 1))

    def sample_x(self, batch_size):
        return torch.rand(size=(batch_size, self.x_dim))


class BoundarySampler:

    def __init__(self, x_dim, boundary_func):
        self.x_dim = x_dim
        self.boundary_func = boundary_func

    def sample_t(self, batch_size):
        return torch.rand(size=(batch_size, 1))

    def sample_x(self, batch_size):
        return torch.rand(size=(batch_size, self.x_dim)).apply_(self.boundary_func)

import torch
from operators import *
from tqdm import tqdm


class FBSDESolver:

    def __init__(self, model_config, fbsde_config):
        self.batch_size = model_config["batch_size"]
        self.num_discretisation_steps = model_config["num_discretisation_steps"]
        self.dt = fbsde_config.terminal_time / model_config["num_discretisation_steps"]
        self.var_dim = fbsde_config.var_dim
        self.sigma = fbsde_config.sigma
        self.gamma = fbsde_config.gamma

    def J(self, *args):
        pass

    def u(self, *args):
        pass

    def train(self, initial_x, iterations):
        X = initial_x
        Y = self.init_Y()
        Z = self.sigma |mdotb| self.init_Z()
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for k in iterations:
            loss = self.train_for_iteration(X, Y, Z)
            yield loss

    def train_for_iteration(self, X, Y, Z):
        dW = self.sigma |mdotb| torch.randn(self.batch_size, self.var_dim)
        for i in range(self.num_discretisation_steps):
            gamma = self.gamma(X, self.dt * i)
            # if bool
            dW = self.sigma |mdotb| torch.randn(self.batch_size, self.var_dim) - dW

    def init_Y(self):
        return torch.zeros(self.batch_size, 1).uniform_().requires_grad()

    def init_Z(self):
        return torch.zeros(self.batch_size, 1).uniform_().requires_grad()

import torch
import torch.nn as nn
from optimisers import OPTIMIZERS
from operators import *
from tqdm import tqdm
from networks import NETWORK_TYPES


class FBSDESolver:

    def __init__(self, model_config, fbsde_config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.batch_size = model_config["batch_size"]
        self.num_discretisation_steps = model_config["num_discretisation_steps"]
        self.dt = fbsde_config.terminal_time / model_config["num_discretisation_steps"]
        self.var_dim = fbsde_config.var_dim

        self.H = fbsde_config.H
        self.sigma = fbsde_config.sigma

        self.C = fbsde_config.C
        self.Gamma = fbsde_config.Gamma
        self.terminal_condition = fbsde_config.terminal_time
        self.inv_D = 1/fbsde_config.D if isinstance(fbsde_config.D, float) or isinstance(fbsde_config.D, int) else torch.inverse(fbsde_config.D)

        self.criterion = torch.nn.MSELoss()
        self.Z_net = NETWORK_TYPES[model_config["network_type"]](input_dim=fbsde_config.var_dim,
                                                                hidden_dim=model_config["hidden_dim"],
                                                                output_dim=fbsde_config.var_dim).to(self.device)
        self.init_Y = nn.Sequential(
            nn.Linear(fbsde_config.var_dim, model_config["hidden_dim"]),
            nn.ELU(),
            nn.Linear(model_config["hidden_dim"], 1)
        )
        self.optimizer = OPTIMIZERS[model_config["optimiser"]](self.Z_net.parameters(), lr=model_config["learning_rate"])

    def J(self, *args):
        pass

    def u(self, *args):
        pass

    def train(self, initial_x, iterations):
        X = initial_x
        Y = self.init_Y(X)
        Z = self.sigma(initial_x, 0) |mdotb| D(Y, X)
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for _ in iterations:
            Y_pred, X_terminal = self.train_for_iteration(X, Y, Z)
            Y_true = self.terminal_condition(X_terminal)
            loss = self.criterion(Y_pred, Y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            yield loss

    def train_for_iteration(self, X, Y, Z):
        dW = self.sigma |mdotb| torch.randn(self.batch_size, self.var_dim)
        for i in range(self.num_discretisation_steps):
            Gamma = self.Gamma(X, self.dt * i)
            sigma = self.sigma(X, self.dt * i)
            # if bool
            U = torch.einsum("bij, bj -> bi", (-self.inv_D @ Gamma), Z)
            dW = sigma |mdotb| torch.randn(self.batch_size, self.var_dim) - dW
            Y = Y - self.C(X)*self.dt \
                + (1/2)*torch.einsum("bi, bij, bj -> b", Z, Gamma, U).unsqueeze(1)*self.dt \
                + torch.einsum("bi, bi -> b", Z, dW).unsqueeze(1)

            X = X + self.H(X, self.dt*i)*self.dt + (sigma |mdotb| (torch.einsum("bij, bj -> bi", Gamma, U)*self.dt + dW))

            Z = self.Z_net(X, self.dt*i)

        return Y, X

from .optimisers import OPTIMIZERS
from .operators import D
import torch
from tqdm import tqdm
from .networks import NETWORK_TYPES
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import numpy as np


class FBSDESolver:

    def __init__(self, model_config, fbsde_config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.batch_size = model_config["batch_size"]
        self.num_discretisation_steps = model_config["num_discretisation_steps"]
        self.dt = (fbsde_config.terminal_time / model_config["num_discretisation_steps"]) * torch.ones(
            (self.batch_size, 1)).to(self.device)
        self.var_dim = fbsde_config.var_dim
        self.init_sampling_func = fbsde_config.init_sampling_func
        self.control_noise = fbsde_config.control_noise

        self.b = fbsde_config.b
        self.h = fbsde_config.h
        self.sigma = fbsde_config.sigma

        self.terminal_condition = fbsde_config.terminal_condition

        self.criterion = torch.nn.MSELoss()
        self.Y_net = NETWORK_TYPES[model_config["network_type"]](input_dim=fbsde_config.var_dim + 1,
                                                                 hidden_dim=model_config["hidden_dim"],
                                                                 output_dim=1).to(self.device)

        self.optimizer = OPTIMIZERS[model_config["optimiser"]](self.Y_net.parameters(),
                                                               lr=model_config["learning_rate"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                           patience=10)

    def J(self, *args):
        vars = [torch.FloatTensor([x]).to(self.device).unsqueeze(0).requires_grad_() for x in args]
        Y_pred = self.Y_net(*vars)
        return Y_pred.detach().flatten().cpu().numpy()

    def D_J(self, *args):
        vars = [torch.FloatTensor([x]).to(self.device).unsqueeze(0).requires_grad_() for x in args]
        Y_pred = self.Y_net(*vars)
        return D(Y_pred, vars[:-1]).flatten().detach().cpu().numpy()

    def simulate_processes(self, num_samples):
        with torch.no_grad():
            X_preds = np.zeros((self.num_discretisation_steps, num_samples, self.var_dim))
            Y_preds = np.zeros((self.num_discretisation_steps, num_samples, 1))
            ts = np.zeros((self.num_discretisation_steps, num_samples, 1))
            dt = self.dt[:num_samples]
            X = self.init_sampling_func(torch.rand(1, self.var_dim).repeat(num_samples, 1)).to(self.device)
            sqrt_dt = torch.sqrt(dt)
            t = Variable(dt * 0)
            Y = self.Y_net(X, t)
            X_preds[0] = X.detach().cpu().numpy()
            Y_preds[0] = Y.detach().cpu().numpy()
            ts[0] = t.detach().cpu().numpy()
            for i in range(1, self.num_discretisation_steps):
                t = Variable(dt * i)
                sigma = self.sigma(X, t)
                dW = sqrt_dt * torch.randn(num_samples, self.var_dim).to(self.device)
                X = X + self.b(X, t) * dt + torch.einsum("bij, bj -> bi", sigma, dW)
                Y = self.Y_net(X, t)
                X_preds[i] = X.detach().cpu().numpy()
                Y_preds[i] = Y.detach().cpu().numpy()
                ts[i] = t.detach().cpu().numpy()

        return X_preds, Y_preds, ts

    def train(self, iterations):
        X = self.init_sampling_func(torch.rand(self.batch_size, self.var_dim)).detach().to(self.device).requires_grad_()
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for _ in iterations:
            loss = self.train_for_iteration(X)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            yield loss.detach().cpu().item()

    def train_for_iteration(self, X):
        loss = 0
        sqrt_dt = torch.sqrt(self.dt)
        t = Variable(self.dt * 0)
        Y = self.Y_net(X, t)
        Z = D(Y, X)
        for i in range(1, self.num_discretisation_steps):
            t = Variable(self.dt * i)
            sigma = self.sigma(X, t)
            K = torch.zeros(self.batch_size, self.var_dim).uniform_(-self.control_noise,
                                                                    self.control_noise).to(self.device).detach()  # torch.einsum("bij, bj -> bi", (-self.inv_D @ M), Z)
            dW = sqrt_dt * torch.randn(self.batch_size, self.var_dim).to(self.device)
            X = X + self.b(X, t) * self.dt + torch.einsum("bij, bj -> bi", sigma, dW) \
                + torch.einsum("bij, bj -> bi", sigma, K) * self.dt
            Y = Y - self.h(X, Y, Z, t) * self.dt \
                + torch.einsum("bi, bi -> b", Z, torch.einsum("bij, bj -> bi", sigma, dW)).unsqueeze(1) \
                + torch.einsum("bi, bij, bj -> b", Z, sigma, K).unsqueeze(1) * self.dt

            X_pred = X.detach().requires_grad_()
            Y_pred = self.Y_net(X_pred, t)
            Z = D(Y_pred, X_pred)
            loss += self.criterion(Y_pred, Y)

        if self.terminal_condition is not None:
            Y_terminal = self.terminal_condition(X)
            loss += self.criterion(Y_pred, Y_terminal)

        return loss


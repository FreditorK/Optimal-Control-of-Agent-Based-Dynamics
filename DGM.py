from sampling import SAMPLING_METHODS
from networks import NETWORK_TYPES
from operators import div, grad, Δ
from torch.optim import Adam
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optimisers import OPTIMIZERS
import torch
import os.path
import torch.nn as nn


class Solver(ABC):
    def __init__(self, model_config):
        """
        Dict:param model_config: Solver configuration
        Dict:param pde_config: PDE to solve
        Tuple:param weights: loss weightings of form (domain, boundary, initial condition)
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.batch_size = model_config["batch_size"]
        self.sampling_method = model_config["sampling_method"]
        self.network_type = model_config["network_type"]
        self.optimiser = model_config["optimiser"]

        self.saveables = {}

    def train(self, iterations):
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for _ in iterations:
            args = self.sample()
            loss = self.backprop_loss(*args)
            yield loss

    @abstractmethod
    def backprop_loss(self, *args):
        ...

    @abstractmethod
    def sample(self):
        ...

    def save(self, path):
        torch.save(self.saveables, path)

    def load(self, path):
        directory, _ = os.path.split(os.path.abspath(__file__))
        path = os.path.join(directory, path)
        checkpoint = torch.load(path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


class DGMSolver(Solver):

    def __init__(self, model_config, pde_config):
        """
        Deep Galerkin PDE Solver
        """
        super().__init__(model_config)
        assert len(pde_config.boundary_cond) == len(pde_config.boundary_func), "Number of boundary " \
                                                                               "conditions does not match" \
                                                                               "number of sampling functions!"
        self.boundary_batch_size = int(self.batch_size / len(pde_config.boundary_func))
        self.domain_sampler = SAMPLING_METHODS[self.sampling_method](pde_config.domain_func, pde_config.var_dim,
                                                                     device=self.device)
        self.boundary_sampler = SAMPLING_METHODS[self.sampling_method](pde_config.boundary_func, pde_config.var_dim,
                                                                       device=self.device)
        self.f_θ = NETWORK_TYPES[self.network_type](input_dim=pde_config.var_dim,
                                                    hidden_dim=model_config["hidden_dim"],
                                                    output_dim=1).to(self.device)

        self.f_θ_optimizer = OPTIMIZERS[self.optimiser](self.f_θ.parameters(), lr=model_config["learning_rate"])

        self.domain_criterion = lambda u, var: \
            model_config["loss_weights"][0] * torch.square(pde_config.equation(u, var))

        self.boundary_criterion = lambda us, vars: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(u, var)) for u, var, bc in zip(us, vars, pde_config.boundary_cond)])

        self.scheduler = ReduceLROnPlateau(self.f_θ_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                           patience=10)

        self.saveables = {
            "f_theta": self.f_θ,
            "f_theta_optimizer": self.f_θ_optimizer
        }

    def u(self, *args):
        with torch.no_grad():
            xs = [torch.FloatTensor([x]).to(self.device).unsqueeze(0) for x in args]
            u = self.f_θ(*xs)
        return u.cpu().numpy().flatten()[0]

    def sample(self):
        domain_var_sample = self.domain_sampler.sample_var(self.batch_size)[0]  # (func(vars), batch, 1)
        boundary_vars_sample = self.boundary_sampler.sample_var(self.boundary_batch_size)  # (subdomain(vars), batch, 1)

        return domain_var_sample, boundary_vars_sample

    def backprop_loss(self, domain_var_sample, boundary_vars_sample):
        domain_u = self.f_θ(*domain_var_sample)
        boundary_us = [self.f_θ(*sample) for sample in boundary_vars_sample]

        boundary_loss = self.boundary_criterion(boundary_us, vars=boundary_vars_sample)
        domain_loss = self.domain_criterion(domain_u, var=domain_var_sample)

        loss = domain_loss.mean() + boundary_loss.mean()

        self.f_θ_optimizer.zero_grad()
        loss.backward()
        self.f_θ_optimizer.step()

        self.scheduler.step(loss)
        self.domain_sampler.update(loss.detach())

        return loss.cpu().detach().flatten()[0].numpy()


class Control_Output(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class DGMPIASolver(Solver):

    def __init__(self, model_config, hbj_config):
        super(DGMPIASolver, self).__init__(model_config)
        F = hbj_config.cost_function
        L = hbj_config.differential_operator
        self.boundary_J_batch_size = int(self.batch_size / len(hbj_config.boundary_func_J))
        self.boundary_u_batch_size = int(self.batch_size / len(hbj_config.boundary_func_u))
        self.domain_sampler = SAMPLING_METHODS[self.sampling_method](hbj_config.domain_func, hbj_config.var_dim,
                                                                     device=self.device)
        self.boundary_sampler_J = SAMPLING_METHODS[self.sampling_method](hbj_config.boundary_func_J, hbj_config.var_dim,
                                                                         device=self.device)
        self.boundary_sampler_u = SAMPLING_METHODS[self.sampling_method](hbj_config.boundary_func_u, 1,
                                                                         device=self.device)
        self.f_θ = NETWORK_TYPES[self.network_type](input_dim=hbj_config.var_dim,
                                                    hidden_dim=model_config["hidden_dim"],
                                                    output_dim=1).to(self.device)  # value_function of (x, t)_u

        self.g_φ = NETWORK_TYPES[self.network_type](input_dim=1,
                                                    hidden_dim=int(model_config["hidden_dim"] / 2),
                                                    output_dim=1).to(self.device)  # control_function of (x, t)_J

        self.f_θ_optimizer = OPTIMIZERS[self.optimiser](self.f_θ.parameters(), lr=model_config["learning_rate"])
        self.g_φ_optimizer = OPTIMIZERS[self.optimiser](self.g_φ.parameters(), lr=model_config["learning_rate"])

        self.differential_criterion = lambda J, u, var: model_config["loss_weights"][0] * torch.square(
            div(J, var[-1]) + L(J, u, var[:-1], var[-1]) + F(u, var[:-1], var[-1]))
        self.first_order_criterion = lambda J, u, var: -(L(J, u, var[:-1], var[-1]) + F(u, var[:-1], var[-1]))
        self.boundary_criterion_J = lambda Js, vars: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(J, var)) for J, var, bc in zip(Js, vars, hbj_config.boundary_cond_J)])
        self.boundary_criterion_u = lambda us, ts: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(u, t)) for u, t, bc in zip(us, ts, hbj_config.boundary_cond_u)])

        self.θ_scheduler = ReduceLROnPlateau(self.f_θ_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                             patience=10)
        self.φ_scheduler = ReduceLROnPlateau(self.g_φ_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                             patience=10)

        self.saveables = {
            "f_theta": self.f_θ,
            "f_theta_optimizer": self.f_θ_optimizer,
            "g_phi": self.g_φ,
            # "g_phi_optimizer": self.g_φ_optimizer
        }

    def J(self, *args):
        with torch.no_grad():
            xs = [torch.FloatTensor([x]).to(self.device).unsqueeze(0) for x in args]
            J = self.f_θ(*xs)
        return J.cpu().numpy().flatten()[0]

    def u(self, t):
        with torch.no_grad():
            t = torch.FloatTensor([t]).to(self.device).unsqueeze(0)
            u = self.g_φ(t)
        return u.cpu().numpy().flatten()[0]

    def sample(self):
        domain_var_sample = self.domain_sampler.sample_var(self.batch_size)[0]  # (func(vars), batch, 1)
        boundary_vars_sample_J = self.boundary_sampler_J.sample_var(
            self.boundary_J_batch_size)  # (subdomain(vars), batch, 1)
        boundary_vars_sample_u = self.boundary_sampler_u.sample_var(
            self.boundary_u_batch_size)  # (subdomain(vars), batch, 1)

        return domain_var_sample, boundary_vars_sample_J, boundary_vars_sample_u

    def backprop_loss(self, domain_var_sample, boundary_vars_sample_J, boundary_vars_sample_u):
        # value
        u = self.g_φ(domain_var_sample[-1])  # u(t)
        J = self.f_θ(*domain_var_sample)  # J(x, t)
        boundary_Js = [self.f_θ(*sample) for sample in boundary_vars_sample_J]  # e.g. terminal conditions

        value_loss = self.differential_criterion(J, u, domain_var_sample).mean() \
                     + self.boundary_criterion_J(boundary_Js, boundary_vars_sample_J).mean()

        self.f_θ_optimizer.zero_grad()
        value_loss.backward()
        self.f_θ_optimizer.step()

        # control
        u = self.g_φ(domain_var_sample[-1])  # u(t)
        boundary_us = [self.g_φ(*sample) for sample in boundary_vars_sample_u]  # e.g. control output restrictions
        J = self.f_θ(*domain_var_sample)  # J(x, t)
        control_loss = self.first_order_criterion(J, u, domain_var_sample).mean() \
                       + self.boundary_criterion_u(boundary_us, boundary_vars_sample_u).mean()

        self.g_φ_optimizer.zero_grad()
        control_loss.backward()
        self.g_φ_optimizer.step()

        self.θ_scheduler.step(value_loss)
        self.φ_scheduler.step(control_loss)

        return value_loss.cpu().detach().flatten()[0].numpy(), control_loss.cpu().detach().flatten()[0].numpy()

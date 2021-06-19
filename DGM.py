from sampling import SAMPLING_METHODS
from networks import FeedForwardNet, DGMNetwork, BVPNetwork
from operators import div, grad, Δ
from torch.optim import Adam
from tqdm import tqdm
from abc import ABC, abstractmethod
from optimisers import Ralamb, RangerLars
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
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
        self.domain_sampler = SAMPLING_METHODS[self.sampling_method](device=self.device)
        self.boundary_sampler = SAMPLING_METHODS[self.sampling_method](device=self.device)
        self.pde_config = pde_config
        self.f_θ = BVPNetwork(input_dim=(pde_config.x_dim, 1),
                              hidden_dim=model_config["hidden_dim"],
                              output_dim=1).to(self.device)

        self.f_θ_optimizer = Adam(self.f_θ.parameters(), lr=model_config["learning_rate"])

        self.domain_criterion = lambda u, x, t: model_config["loss_weights"][0] * torch.square(
            pde_config.equation(u, x, t)).mean()
        if pde_config.boundary_cond is None:
            self.boundary_criterion = lambda u, x, t: 0
        else:
            self.boundary_criterion = lambda u, x, t: \
                model_config["loss_weights"][1] * torch.square(u - pde_config.boundary_cond(x, t)).mean()
        if pde_config.init_datum is None:
            self.init_criterion = lambda u, x: 0
        else:
            self.init_criterion = lambda u, x: \
                model_config["loss_weights"][2] * torch.square(u - pde_config.init_datum(x)).mean()

        self.scheduler = ReduceLROnPlateau(self.f_θ_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                           patience=10)

        self.saveables = {
            "f_theta": self.f_θ,
            "f_theta_optimizer": self.f_θ_optimizer
        }

    def u(self, *args, t):
        with torch.no_grad():
            x = torch.FloatTensor(args).to(self.device).unsqueeze(0)
            t = torch.FloatTensor([t]).to(self.device).unsqueeze(0)
            u = self.f_θ(x, t)
        return u.cpu().numpy().flatten()[0]

    def sample(self):
        domain_t_sample = self.domain_sampler.sample_var(self.batch_size, 1).requires_grad_()
        boundary_t_sample = self.boundary_sampler.sample_var(self.batch_size, 1).requires_grad_()
        t_0_sample = torch.zeros_like(boundary_t_sample).to(self.device).requires_grad_()

        domain_x_sample = self.domain_sampler.sample_var(self.batch_size, self.pde_config.x_dim,
                                                         self.pde_config.domain_func).requires_grad_()
        boundary_x_sample = self.boundary_sampler.sample_var(self.batch_size, self.pde_config.x_dim,
                                                             self.pde_config.boundary_func).requires_grad_()
        x_0_sample = self.domain_sampler.sample_var(self.batch_size, self.pde_config.x_dim,
                                                    self.pde_config.init_func).requires_grad_()

        return domain_t_sample, boundary_t_sample, t_0_sample, \
               domain_x_sample, boundary_x_sample, x_0_sample

    def backprop_loss(self, domain_t_sample, boundary_t_sample, t_0_sample, domain_x_sample, boundary_x_sample,
                      x_0_sample):
        domain_u = self.f_θ(domain_x_sample, domain_t_sample)
        boundary_u = self.f_θ(boundary_x_sample, boundary_t_sample)
        domain_u_0 = self.f_θ(x_0_sample, t_0_sample)

        boundary_loss = self.boundary_criterion(u=boundary_u, x=boundary_x_sample, t=boundary_t_sample)
        domain_loss = self.init_criterion(u=domain_u_0, x=x_0_sample) \
                      + self.domain_criterion(u=domain_u, x=domain_x_sample, t=domain_t_sample)

        loss = domain_loss + boundary_loss

        self.f_θ_optimizer.zero_grad()
        loss.backward()
        self.f_θ_optimizer.step()

        self.scheduler.step(loss)

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
        self.F = hbj_config.cost_function
        G = hbj_config.terminal_cost
        L = hbj_config.differential_operator
        self.hbj_config = hbj_config
        self.sampler = SAMPLING_METHODS[self.sampling_method](device=self.device)
        self.f_θ = FeedForwardNet(input_dim=(sum(hbj_config.var_dims), 1),
                                  hidden_dim=model_config["hidden_dim"],
                                  output_dim=1).to(self.device)  # value_function of (x, t)_u
        self.g_φ = lambda t: (hbj_config.μ - hbj_config.r) / (hbj_config.σ ** 2 * (1 - hbj_config.γ) * hbj_config.γ)
        '''
        self.g_φ = nn.Sequential(FeedForwardNet(input_dim=[1],
                                  hidden_dim=model_config["hidden_dim"],
                                  output_dim=1).to(self.device),
                                 Control_Output(hbj_config.control_output)
                                 )# control_function of (x, t)_J
        '''

        self.f_θ_optimizer = Adam(self.f_θ.parameters(), lr=model_config["learning_rate"])
        # self.g_φ_optimizer = Adam(self.g_φ.parameters(), lr=model_config["learning_rate"])

        self.differential_criterion = lambda J, u, x, t: model_config["loss_weights"][0] * torch.square(
            div(J, t) + L(J, u, x, t) + F(u, x, t)).mean()
        self.terminal_criterion = lambda J, x: model_config["loss_weights"][1] * torch.square(
            J - G(x)).mean()
        self.first_order_criterion = lambda J, u, x, t: - model_config["batch_size"] * model_config["loss_weights"][
            2] * (L(J, u, x, t) + F(u, x, t)).mean()

        self.saveables = {
            "f_theta": self.f_θ,
            "f_theta_optimizer": self.f_θ_optimizer,
            "g_phi": self.g_φ,
            # "g_phi_optimizer": self.g_φ_optimizer
        }

    def u(self, t):
        with torch.no_grad():
            t = torch.FloatTensor([t]).to(self.device).unsqueeze(0)
            u = self.g_φ(t)
        return u.cpu().numpy().flatten()[0]

    def J(self, x, t):
        with torch.no_grad():
            x = torch.FloatTensor([x]).to(self.device).unsqueeze(0)
            t = torch.FloatTensor([t]).to(self.device).unsqueeze(0)
            j = self.f_θ(x, t)
        return j.cpu().numpy().flatten()[0]

    def sample(self):
        t_sample = self.sampler.sample_var(self.batch_size, 1).requires_grad_()
        T_sample = torch.ones_like(t_sample).to(self.device).requires_grad_()

        vars_samples = []
        vars_T_samples = []
        for var_dim, sampling_func in zip(self.hbj_config.var_dims, self.hbj_config.sampling_funcs):
            vars_samples.append(self.sampler.sample_var(self.batch_size, var_dim, sampling_func).requires_grad_())
            vars_T_samples.append(self.sampler.sample_var(self.batch_size, var_dim, sampling_func).requires_grad_())

        return t_sample, T_sample, vars_samples, vars_T_samples

    def backprop_loss(self, t_sample, T_sample, vars_samples, vars_T_samples):
        # value
        u = self.g_φ(t_sample)  # .detach()
        J = self.f_θ(*vars_samples, t_sample)
        J_T = self.f_θ(*vars_T_samples, T_sample)

        value_loss = self.differential_criterion(J, u, vars_samples, t_sample) \
                     + self.terminal_criterion(J_T, T_sample)

        self.f_θ_optimizer.zero_grad()
        value_loss.backward()
        self.f_θ_optimizer.step()

        # control
        u = self.g_φ(t_sample)
        J = self.f_θ(*vars_samples, t_sample)
        control_loss = self.first_order_criterion(J, u, vars_samples, t_sample)

        # self.g_φ_optimizer.zero_grad()
        # control_loss.backward()
        # self.g_φ_optimizer.step()

        return value_loss.cpu().detach().flatten()[0].numpy(), control_loss.cpu().detach().flatten()[0].numpy()

from .sampling import SAMPLING_METHODS
from .networks import NETWORK_TYPES
from .operators import div, D
from .optimisers import OPTIMIZERS
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os.path


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
            print("Using CPU!")
            self.device = torch.device("cpu")

        self.sampling_method = model_config["sampling_method"]
        self.sampling_method_boundary = model_config["sampling_method"]
        if "sampling_method_boundary" in model_config:
            self.sampling_method_boundary = model_config["sampling_method_boundary"]
        self.network_type = model_config["network_type"]
        self.optimiser = model_config["optimiser"]

        self.saveables = {}

    def train(self, iterations):
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for i in iterations:
            args = self.sample()
            loss = self.backprop_loss(i, *args)
            yield loss

    @abstractmethod
    def backprop_loss(self, *args):
        ...

    @abstractmethod
    def sample(self):
        ...

    def save(self, path):
        directory, _ = os.path.split(os.path.abspath(__file__))
        path = os.path.join(directory, path)
        torch.save(self.saveables, path)

    def load(self, path):
        directory, _ = os.path.split(os.path.abspath(__file__))
        path = os.path.join(directory, path)
        checkpoint = torch.load(path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


class DeepPDESolver(Solver):

    def __init__(self, model_config, pde_config):
        """
        Deep Galerkin PDE Solver
        """
        super().__init__(model_config)
        assert len(pde_config.boundary_cond) == len(pde_config.boundary_func), "Number of boundary " \
                                                                               "conditions does not match" \
                                                                               "number of sampling functions!"
        self.domain_sampler = SAMPLING_METHODS[self.sampling_method](pde_config.domain_func, pde_config.var_dim,
                                                                     self.device, pde_config.__class__.__name__)
        self.boundary_sampler = SAMPLING_METHODS[self.sampling_method_boundary + "_bound"](pde_config.boundary_func, pde_config.var_dim,
                                                                       self.device, pde_config.__class__.__name__, self.domain_sampler)
        self.f_?? = NETWORK_TYPES[self.network_type](input_dim=pde_config.var_dim,
                                                    hidden_dim=model_config["hidden_dim"],
                                                    output_dim=pde_config.sol_dim).to(self.device)

        self.f_??_optimizer = OPTIMIZERS[self.optimiser](self.f_??.parameters(), lr=model_config["learning_rate"])

        self.domain_criterion = lambda us, vars: \
            model_config["loss_weights"][0] * sum(
                [torch.square(pde_config.equation(u, var)).mean() for u, var in zip(us, vars)])

        self.boundary_criterion = lambda us, vars: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(u, var)).mean() for u, var, bc in zip(us, vars, pde_config.boundary_cond)])

        self.scheduler = ReduceLROnPlateau(self.f_??_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                           patience=10)

        self.saveables = {
            "f_theta": self.f_??,
            "f_theta_optimizer": self.f_??_optimizer
        }

        self.ls = []

    def u(self, *args):
        with torch.no_grad():
            xs = [torch.FloatTensor([x]).to(self.device).unsqueeze(0) for x in args]
            u = self.f_??(*xs)
        return u.cpu().numpy().flatten()

    def D_u(self, *args):
        xs = [torch.FloatTensor([x]).to(self.device).unsqueeze(0).requires_grad_() for x in args]
        u = self.f_??(*xs)
        return D(u, xs[:-1]).detach().cpu().numpy().flatten()

    def sample(self):
        domain_var_sample = self.domain_sampler.sample_var()  # (func(vars), batch, 1)
        boundary_vars_sample = self.boundary_sampler.sample_var()  # (subdomain(vars), batch, 1)

        return domain_var_sample, boundary_vars_sample

    def backprop_loss(self, i, domain_var_sample, boundary_vars_sample):
        domain_u = [self.f_??(*sample) for sample in domain_var_sample]
        boundary_us = [self.f_??(*sample) for sample in boundary_vars_sample]

        boundary_loss = self.boundary_criterion(boundary_us, vars=boundary_vars_sample)
        domain_loss = self.domain_criterion(domain_u, vars=domain_var_sample)

        loss = domain_loss + boundary_loss

        self.domain_sampler.update(domain_u, domain_var_sample, i)

        self.f_??_optimizer.zero_grad()
        loss.backward()
        self.f_??_optimizer.step()

        self.scheduler.step(loss)

        return loss.cpu().detach().flatten()[0].numpy()


class DGMPIASolver(Solver):

    def __init__(self, model_config, hbj_config):
        super(DGMPIASolver, self).__init__(model_config)
        F = hbj_config.cost_function
        L = hbj_config.differential_operator
        self.delay_control = model_config["delay_control"]
        self.control_vars = hbj_config.control_vars
        self.domain_sampler = SAMPLING_METHODS[self.sampling_method](hbj_config.domain_func, hbj_config.var_dim_J,
                                                                     self.device, hbj_config.__class__.__name__)
        self.boundary_sampler_J = SAMPLING_METHODS[self.sampling_method_boundary + "_bound"](hbj_config.boundary_func_J,
                                                                                           hbj_config.var_dim_J,
                                                                                           self.device,
                                                                                           hbj_config.__class__.__name__,
                                                                                           self.domain_sampler)

        self.boundary_sampler_u = SAMPLING_METHODS[self.sampling_method_boundary](hbj_config.boundary_func_u,
                                                                                  len(self.control_vars),
                                                                                  self.device, hbj_config.__class__.__name__,
                                                                                  self.domain_sampler)

        self.f_?? = NETWORK_TYPES[self.network_type](input_dim=hbj_config.var_dim_J,
                                                    hidden_dim=model_config["hidden_dim_J"],
                                                    output_dim=1).to(self.device)  # value_function of (x, t)_J

        self.g_?? = NETWORK_TYPES[self.network_type](input_dim=len(self.control_vars),
                                                    hidden_dim=model_config["hidden_dim_u"],
                                                    output_dim=hbj_config.sol_dim).to(
            self.device)  # control_function of (x, t)_u

        self.alpha = torch.tensor(model_config["alpha_noise"], requires_grad=False)

        self.f_??_optimizer = OPTIMIZERS[self.optimiser](self.f_??.parameters(), lr=model_config["learning_rate"])
        self.g_??_optimizer = OPTIMIZERS[self.optimiser](self.g_??.parameters(), lr=model_config["learning_rate"])

        self.differential_criterion = lambda J, u, var: model_config["loss_weights"][0] * torch.square(
            div(J, var[-1]) + L(J, u, var) + F(u, var)).mean()
        self.first_order_criterion = lambda J, u, var: -(L(J, u, var) + F(u, var)).mean()
        self.boundary_criterion_J = lambda Js, vars: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(J, var)).mean() for J, var, bc in zip(Js, vars, hbj_config.boundary_cond_J)])
        self.boundary_criterion_u = lambda us, vars: \
            model_config["loss_weights"][1] * sum(
                [torch.square(bc(u, var)).mean() for u, var, bc in zip(us, vars, hbj_config.boundary_cond_u)])

        self.??_scheduler = ReduceLROnPlateau(self.f_??_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                             patience=10)
        self.??_scheduler = ReduceLROnPlateau(self.g_??_optimizer, 'min', factor=model_config["lr_decay"], min_lr=1e-10,
                                             patience=10)
        self.saveables = {
            "f_theta": self.f_??,
            "f_theta_optimizer": self.f_??_optimizer,
            "g_phi": self.g_??,
            "g_phi_optimizer": self.g_??_optimizer
        }

    def J(self, *args):
        with torch.no_grad():
            vars = [torch.FloatTensor([x]).to(self.device).unsqueeze(0) for x in args]
            J = self.f_??(*vars)
        return J.cpu().numpy().flatten()

    def u(self, *args):
        with torch.no_grad():
            vars = [torch.FloatTensor([x]).to(self.device).unsqueeze(0) for x in args]
            u = self.g_??(*vars)
        return u.cpu().numpy().flatten()

    def sample(self):
        domain_var_sample = self.domain_sampler.sample_var()[0]  # (func(vars), batch, 1)
        boundary_vars_sample_J = self.boundary_sampler_J.sample_var()  # (subdomain(vars), batch, 1)
        boundary_vars_sample_u = self.boundary_sampler_u.sample_var()  # (subdomain(vars), batch, 1)

        return domain_var_sample, boundary_vars_sample_J, boundary_vars_sample_u

    def backprop_loss(self, i, domain_var_sample, boundary_vars_sample_J, boundary_vars_sample_u):
        # value
        u = self.g_??(*[domain_var_sample[i] for i in self.control_vars])  # u(t)
        u += (self.alpha * torch.randn_like(u)).clamp(-0.5, 0.5)
        J = self.f_??(*domain_var_sample)  # J(x, t)
        boundary_Js = [self.f_??(*sample) for sample in boundary_vars_sample_J]  # e.g. terminal conditions

        value_loss = self.differential_criterion(J, u, domain_var_sample) \
                     + self.boundary_criterion_J(boundary_Js, boundary_vars_sample_J)

        self.f_??_optimizer.zero_grad()
        value_loss.backward()
        self.f_??_optimizer.step()
        self.??_scheduler.step(value_loss)

        domain_var_sample = [var.detach().requires_grad_() for var in domain_var_sample]  # resets gradients to zero

        # control
        u = self.g_??(*[domain_var_sample[i] for i in self.control_vars])  # u(t)
        boundary_us = [self.g_??(*sample) for sample in boundary_vars_sample_u]  # e.g. control output restrictions
        J = self.f_??(*domain_var_sample)  # J(x, t)
        control_loss = (self.first_order_criterion(J, u, domain_var_sample)
                        + self.boundary_criterion_u(boundary_us, boundary_vars_sample_u))  # *self.alpha

        if i % self.delay_control == 0:
            self.g_??_optimizer.zero_grad()
            control_loss.backward()
            self.g_??_optimizer.step()
            self.??_scheduler.step(control_loss)

            self.alpha *= 0.99

        return value_loss.cpu().detach().flatten()[0].numpy(), control_loss.cpu().detach().flatten()[0].numpy()

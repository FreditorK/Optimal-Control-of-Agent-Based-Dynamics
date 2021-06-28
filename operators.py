from torch.autograd import grad
import torch

class Infix:
    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


def m_func(x, y):
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
        return torch.einsum("bi, ij -> bj", x, y)
    y = torch.cat(y, dim=-1)
    return torch.einsum("ij, bj -> bi", x, y)


def v_func(x, y):
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
    if isinstance(y, list):
        y = torch.cat(y, dim=-1)

    return torch.einsum("bi, bi -> b", x, y).unsqueeze(1)


mdot = Infix(m_func)
vdot = Infix(v_func)


def D(u, vars):
    u_grad = grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u))
    return torch.cat(u_grad, dim=-1)


def div(u, vars):
    """
    Divergence operator
    Tensor (batch, 1):param u: function to differentiate
    Tensor (batch, x_dim):param vars: variables to differentiate in respect to
    Tensor (batch, 1):return: divergence div(u) w.r.t. vars
    """
    u_grad_sum = grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u))
    return sum(u_grad_sum)


def Δ(u, vars):
    """
    Laplacian operator
    Tensor (batch, 1):param u: function to differentiate
    Tensor (batch, x_dim):param vars: variables to differentiate in respect to
    Tensor (batch, 1):return: laplacian Δu w.r.t. vars
    """
    u_grad_sum = sum(grad(u, vars, create_graph=True, grad_outputs=torch.ones_like(u)))
    u_hess_diag = sum(grad(u_grad_sum, vars, create_graph=True, grad_outputs=torch.ones_like(u_grad_sum)))
    u_lap = torch.sum(u_hess_diag, dim=-1).unsqueeze(1)
    return u_lap

from torch.autograd import grad
import torch


def div(u, vars):
    """
    Divergence operator
    Tensor (batch, 1):param u: function to differentiate
    Tensor (batch, x_dim):param vars: variables to differentiate in respect to
    Tensor (batch, 1):return: divergence div(u) w.r.t. vars
    """
    u_grad = grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    return u_grad


def Δ(u, vars):
    """
    Laplacian operator
    Tensor (batch, 1):param u: function to differentiate
    Tensor (batch, x_dim):param vars: variables to differentiate in respect to
    Tensor (batch, 1):return: laplacian Δu w.r.t. vars
    """
    u_grad = torch.autograd.grad(u, vars, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_hess_diag = torch.autograd.grad(u_grad, vars, create_graph=True, grad_outputs=torch.ones_like(u_grad))[0]
    u_lap = torch.sum(u_hess_diag, dim=-1).unsqueeze(1)
    return u_lap
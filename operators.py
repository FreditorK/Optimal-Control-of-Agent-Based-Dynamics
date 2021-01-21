from torch.autograd import grad
import torch


def div(u, vars):
    '''
    Tensor:param u:
    Tensor:param vars:
    Tensor:return: sum of gradients/divergence of u w.r.t vars
    '''
    # grad computes J*v, where J is the Jacobian of u and v is grad_outputs
    return torch.sum(grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u))[0], dim=-2)


def Δ(u, vars):
    u_grad = torch.autograd.grad(u, vars, create_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_hess_diag = torch.autograd.grad(u_grad, vars, create_graph=True, grad_outputs=torch.ones_like(u_grad))[0]
    return torch.sum(u_hess_diag, dim=-2)

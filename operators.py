from torch.autograd import grad
import torch


def d(u, var):
    return grad(outputs=u, inputs=var, grad_outputs=torch.ones_like(u), retain_graph=True)[0]


def dd(u, var_1):
    du = grad(outputs=u, inputs=var_1, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    ddu = grad(outputs=du, inputs=var_1, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
    return ddu

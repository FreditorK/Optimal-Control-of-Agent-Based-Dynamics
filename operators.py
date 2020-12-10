from torch.autograd import grad
import torch


def d(u, *vars):
    if vars[0] == ():
        return u
    return d(grad(outputs=u, inputs=vars[0], grad_outputs=torch.ones_like(u), retain_graph=True)[0], vars[1:])

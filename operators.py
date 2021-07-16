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


def m_func_b(x, y):
    """
    Takes the right-side vector-matrix dot-product batch-wise
    """
    if isinstance(y, list):
        y = torch.cat(y, dim=-1)
    return torch.einsum("ij, bj -> bi", x, y)


def b_func_m(x, y):
    """
    Takes the left-side vector-matrix dot-product batch-wise
    """
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
    return torch.einsum("bi, ij -> bj", x, y)


def m_func_m(x, y):
    """
    Takes matrix dot-product
    """
    return x @ y


def b_func_b(x, y):
    """
    Takes batch-wise dot-product
    """
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
    if isinstance(y, list):
        y = torch.cat(y, dim=-1)

    return torch.einsum("bi, bi -> b", x, y).unsqueeze(1)


def minus_(x, y):
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
    if isinstance(y, list):
        y = torch.cat(y, dim=-1)

    return x - y


def plus_(x, y):
    if isinstance(x, list):
        x = torch.cat(x, dim=-1)
    if isinstance(y, list):
        y = torch.cat(y, dim=-1)

    return x + y


def cat(x):
    return torch.cat(x, dim=-1)


mdotb = Infix(m_func_b)
bdotm = Infix(b_func_m)
mdotm = Infix(m_func_m)
bdotb = Infix(b_func_b)
m = Infix(minus_)
p = Infix(plus_)


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
    u_grad = grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u))
    return sum(u_grad)


def Δ(u, vars):
    """
    Laplacian operator
    Tensor (batch, 1):param u: function to differentiate
    Tensor (batch, x_dim):param vars: variables to differentiate in respect to
    Tensor (batch, 1):return: laplacian Δu w.r.t. vars
    """
    if isinstance(vars, list):
        return sum([div(div(u, vars[i]), vars[i]) for i in range(len(vars))])
    return div(div(u, vars), vars)


def H(u, vars):
    hessian_rows = []
    u_grads = list(grad(outputs=u, inputs=vars, create_graph=True, grad_outputs=torch.ones_like(u)))
    for u_grad in u_grads:
        hessian_rows.append(
            torch.cat(grad(u_grad, vars, create_graph=True, grad_outputs=torch.ones_like(u_grad)), dim=-1).unsqueeze(
                -1))
    return torch.cat(hessian_rows, dim=-1)

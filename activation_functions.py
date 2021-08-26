import torch
import torch.nn as nn


class dSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x) * (x + torch.exp(x) + 1) / (torch.exp(x) + 1) ** 2


class PSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)

    def forward(self, x):
        return self.alpha * x / (1 + torch.exp(-self.beta * x))


# Bipolar funnctions: useless
class SoftExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)

    def forward(self, x):
        if self.alpha.item() == 0:
            return x**1
        if self.alpha.item() < 0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha
        return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


class brelu_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x[1::2] = x[1::2].clamp(min=0)
        x[::2] = -(-x[::2]).clamp(min=0)
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        grads = grad_outputs.clone()
        x = ctx.saved_tensors[0]
        grads[1::2] = (x[1::2] >= 0).float() * grads[1::2]
        grads[::2] = (x[::2] < 0).float() * grads[::2]
        return grads


class mean_zero_silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x[1::2] = x[1::2] / (1 + torch.exp(-x[1::2]))
        x[::2] = x[::2]/(1+torch.exp(x[::2]))
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        grads = grad_outputs.clone()
        x = ctx.saved_tensors[0]
        input_shape = x.shape[0]
        even_indices = [i for i in range(0, input_shape, 2)]
        odd_indices = [i for i in range(1, input_shape, 2)]
        grads[1::2] = torch.exp(x[odd_indices]) * (x[odd_indices] + torch.exp(x[odd_indices]) + 1) / (torch.exp(x[odd_indices]) + 1) ** 2 * grads[odd_indices]
        grads[::2] = (1-torch.exp(x[even_indices])*(x[even_indices] - 1))/(torch.exp(x[even_indices])+1)**2 * grads[even_indices]
        return grads


class MSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mean_zero_silu.apply(x)


class PBReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)

    def forward(self, x):
        return self.alpha * torch.sinh(-brelu_function.apply(x))

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


class SoftExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)

    def forward(self, x):
        if self.alpha.item() == 0:
            return x
        if self.alpha.item() < 0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha
        return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


class brelu_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x = torch.cat(args, dim=-1)
        ctx.save_for_backward(x)
        x[1::2] = x[1::2].clamp(min=0)
        x[::2] = -(-x[::2]).clamp(min=0)
        return x

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = torch.cat(grad_outputs, dim=-1)
        x = ctx.saved_tensors[0]
        grads[1::2] = (x[1::2] >= 0).float() * grads[1::2]
        grads[::2] = (x[::2] < 0).float() * grads[::2]
        return grads


class PBReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1).cuda(), requires_grad=True)

    def forward(self, x):
        return self.alpha * torch.sinh(-brelu_function.apply(x))

import torch.nn as nn
import torch
from itertools import product
from torchdyn.models import NeuralDE


class DENSNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = GRUNetwork(input_dim, hidden_dim, output_dim)

    def forward(self, *vars):
        return torch.exp(self.net(*vars))


class DGMNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGMNetwork, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU()
        )

        self.layer_1 = DGMLayer(input_dim, hidden_dim)
        self.layer_2 = DGMLayer(input_dim, hidden_dim)
        self.layer_3 = DGMLayer(input_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        S = self.init_layer(xt)
        S = self.layer_1(xt, S)
        S = self.layer_2(xt, S)
        S = self.layer_3(xt, S)
        return self.output_layer(S)


class DGMLayer(nn.Module):

    def __init__(self, input_dim, S_dim):
        super(DGMLayer, self).__init__()
        self.Z_layer = nn.Linear(input_dim, S_dim)
        self.Z_prev = nn.Linear(S_dim, S_dim)
        self.Z_activation = nn.Tanh()

        self.G_layer = nn.Linear(input_dim, S_dim)
        self.G_prev = nn.Linear(S_dim, S_dim)
        self.G_activation = nn.Tanh()

        self.R_layer = nn.Linear(input_dim, S_dim)
        self.R_prev = nn.Linear(S_dim, S_dim)
        self.R_activation = nn.Tanh()

        self.H_layer = nn.Linear(input_dim, S_dim)
        self.H_prev = nn.Linear(S_dim, S_dim)
        self.H_activation = nn.Tanh()

    def forward(self, x, S):
        Z = self.Z_activation(self.Z_layer(x) + self.Z_prev(S))
        G = self.G_activation(self.G_layer(x) + self.G_prev(S))
        R = self.R_activation(self.R_layer(x) + self.R_prev(S))
        H = self.H_activation(self.H_layer(x) + self.H_prev(S * R))

        return (1 - G) * H + Z * S


class GRUNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()

        self.init_denominator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
        )

        self.denominator_1 = nn.GRUCell(input_dim, hidden_dim)
        self.denominator_2 = nn.GRUCell(input_dim, hidden_dim)

        self.init_numerator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
        )

        self.numerator_1 = nn.GRUCell(input_dim, hidden_dim)
        self.numerator_2 = nn.GRUCell(input_dim, hidden_dim)
        self.numerator_3 = nn.GRUCell(input_dim, hidden_dim)

        self.numerator = nn.Linear(hidden_dim, output_dim)
        self.denominator = nn.Linear(hidden_dim, output_dim)

        self.interpolator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        N = self.init_numerator(xt)
        N = self.numerator_1(xt, N)
        N = self.numerator_2(xt, N)
        N = self.numerator_3(xt, N)
        N_f = self.numerator(N)

        D = self.init_denominator(xt)
        D = self.denominator_1(xt, D)
        D = self.denominator_2(xt, D)

        I = self.interpolator(xt)

        return I * N_f + (1 - I) * torch.div(N_f, torch.exp(self.denominator(D)))


class RESNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RESNetwork, self).__init__()
        self.net = nn.Sequential(
            ResLayer(input_dim, hidden_dim, 2 * hidden_dim),
            ResLayer(2 * hidden_dim, 4 * hidden_dim, 2 * hidden_dim),
            ResLayer(2 * hidden_dim, hidden_dim, output_dim)
        )

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        return self.net(xt)


class ResLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class MININetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MININetwork, self).__init__()
        self.y_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        y = self.y_net(xt)
        return y


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.y_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        weights = self.y_net(xt)
        return weights + self.shortcut(xt)


class MeanNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MeanNetwork, self).__init__()
        self.y_net = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shortcut = nn.Linear(input_dim+1, output_dim)

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        mean = torch.mean(xt, dim=1, keepdim=True)
        xt = torch.cat((xt, mean), dim=1)
        weights = self.y_net(xt)
        return weights + self.shortcut(xt)



NETWORK_TYPES = {
    "DGM": DGMNetwork,
    "GRU": GRUNetwork,
    "DENS": DENSNetwork,
    "RES": RESNetwork,
    "MINI": MININetwork,
    "FF": FeedForwardNetwork,
    "MEAN": MeanNetwork
}

"""
class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.y_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 36),
        )

        self.zero_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 36),
            nn.Sigmoid()
        )

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        weights = self.y_net(xt)
        zeroed = torch.round(self.zero_net(xt))
        return torch.sum(zeroed*weights*torch.cat([vars[0]**i * vars[1]**j for i, j in product(range(6), range(6))], dim=1), dim=-1)
"""

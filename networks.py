import torch.nn as nn
import torch
from torchdyn.models import NeuralDE


class BVPNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Tuple:param input_dim: (x_dim, t_dim)
        int:param hidden_dim: number of hidden nodes
        int:param output_dim: u_dim
        """
        super().__init__()

        self.spatial_net = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.domain_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.final_layer = nn.Linear(output_dim * 2, output_dim)

    def forward(self, *vars):
        domain = self.domain_net(torch.cat(vars, dim=1))
        spatial = self.spatial_net(torch.cat(vars[:-1], dim=1))
        return self.final_layer(torch.cat((domain, spatial), dim=1))


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
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU()
        )

        self.layer_1 = nn.GRUCell(input_dim, hidden_dim)
        self.layer_2 = nn.GRUCell(input_dim, hidden_dim)
        self.layer_3 = nn.GRUCell(input_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        S = self.init_layer(xt)
        S = self.layer_1(xt, S)
        S = self.layer_2(xt, S)
        S = self.layer_3(xt, S)
        return self.output_layer(S)


class ResNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResNetwork, self).__init__()
        self.net = nn.Sequential(
            ResLayer(input_dim, hidden_dim, 2*hidden_dim),
            ResLayer(2*hidden_dim, 4*hidden_dim, 2*hidden_dim),
            ResLayer(2*hidden_dim, hidden_dim, output_dim)
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


NETWORK_TYPES = {
    "BVP": BVPNetwork,
    "DGM": DGMNetwork,
    "GRU": GRUNetwork,
    "Res": ResNetwork
}

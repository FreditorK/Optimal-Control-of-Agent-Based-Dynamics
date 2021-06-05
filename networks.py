import torch.nn as nn
import torch


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Tuple:param input_dim: (x_dim, t_dim)
        int:param hidden_dim: number of hidden nodes
        int:param output_dim: u_dim
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(input_dim), hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, *vars):
        xt = torch.cat(vars, dim=1)
        output = self.net(xt)

        return output

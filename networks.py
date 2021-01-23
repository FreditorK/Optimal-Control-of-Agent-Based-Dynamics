import torch.nn as nn
import torch


class f_theta(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(input_dim), hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        xt = torch.cat((x, t), dim=1)
        output = self.net(xt)

        return output

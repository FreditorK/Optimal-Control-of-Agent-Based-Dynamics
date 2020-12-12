from sampling import Sampler
from networks import f_theta
from torch.optim import Adam, SGD
from torch.autograd import Variable
from tqdm import tqdm
import torch
import os.path


class DGMSolver:

    def __init__(self, model_config, pde_config, weights=(0.5, 0.5)):
        self.batch_size = model_config["batch_size"]
        self.sampler = Sampler(pde_config["x_dim"])
        self.f_theta = f_theta(input_dim=(pde_config["x_dim"], 1), hidden_dim=model_config["hidden_dim"], output_dim=1)
        self.criterion = lambda u, x, t: weights[0] * torch.square(pde_config["equation"](u, x, t)).mean() + weights[
            1] * torch.square(pde_config["init_datum"](u, x)).mean()
        self.optimizer = Adam(self.f_theta.parameters(), lr=model_config["learning_rate"])
        self.saveables = {
            "f_theta": self.f_theta,
            "optimizer": self.optimizer
        }

    def train(self, iterations, plot_loss=True):
        iterations = tqdm(range(iterations), leave=True, unit=" it")
        for t in iterations:
            t_sample = self.sampler.sample_t(self.batch_size).requires_grad_()
            x_sample = self.sampler.sample_x(self.batch_size).requires_grad_()
            u = self.f_theta(x_sample, t_sample)
            loss = self.criterion(u=u, x=x_sample, t=t_sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if plot_loss and t % 10 == 0:
                yield loss.detach().flatten()[0].numpy()

    def u(self, x, t):
        with torch.no_grad():
            x = torch.FloatTensor([x]).unsqueeze(0)
            t = torch.FloatTensor([t]).unsqueeze(0)
            u = self.f_theta(x, t)
        return u.numpy().flatten()[0]

    def save(self, path):
        torch.save(self.saveables, path)

    def load(self, path):
        directory, _ = os.path.split(os.path.abspath(__file__))
        path = os.path.join(directory, path)
        checkpoint = torch.load(path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())
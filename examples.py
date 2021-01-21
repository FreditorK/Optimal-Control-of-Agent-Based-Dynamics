from operators import div, Δ
from DGM import DGMSolver
import argparse
import matplotlib.pyplot as plt
import numpy as np

#  Network configurations
MODEL_CONFIG_1 = {
    "batch_size": 128, # minimum batch size is two because of split
    "hidden_dim": 256,
    "learning_rate": 1e-6
}

#  PDE configurations
BURGERS_CONFIG = {
    "x_dim": 1,
    "equation": lambda u, x, t: div(u, t) + u * div(u, x),
    "boundary_cond": lambda u, x, t: u*0,
    "boundary_func": lambda x: x,
    "init_datum": lambda u, x: x - u
}
burgers_sol = lambda x, t: x / (1 + t)

VISCOUS_BURGERS_CONFIG = {
    "x_dim": 1,
    "equation": lambda u, x, t: div(u, t) + u * div(u, x) - 0.5 * Δ(u, x),
    "boundary_cond": lambda u, x, t: u,
    "boundary_func": lambda x: 1.0 if x > 0.0 else -1.0,
    "init_datum": lambda u, x: x - u
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSc Project, Frederik Kelbel')
    parser.add_argument('--it', nargs="?", type=int, default=3000, help='number of iterations')
    args = parser.parse_args()

    solver = DGMSolver(MODEL_CONFIG_1, VISCOUS_BURGERS_CONFIG)
    losses = list(solver.train(args.it))
    plt.plot(np.convolve(losses, np.ones(10), 'valid') / 10)
    plt.show()
    '''xs = np.array(range(200))/100 - 1.0
    y = burgers_sol(xs, 1.0)
    y_pred = [solver.u(x, 1.0) for x in xs]
    plt.plot(xs, y)
    plt.plot(xs, y_pred)
    plt.show()'''

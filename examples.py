from operators import d, dd
from DGM import DGMSolver
import argparse
import matplotlib.pyplot as plt

#  Network configurations
MODEL_CONFIG_1 = {
    "batch_size": 128,
    "hidden_dim": 256,
    "learning_rate": 1e-4
}

#  PDE configurations
BURGERS_CONFIG = {
    "x_dim": 1,
    "equation": lambda u, x, t: d(u, t) + u * d(u, x),
    "boundary_cond": lambda u, x, t: u*0,
    "boundary_func": lambda random, x: x,
    "init_datum": lambda u, x: x - u
}
burgers_sol = lambda x, t: x / (1 + t)

VISCOUS_BURGERS_CONFIG = {
    "x_dim": 1,
    "equation": lambda u, x, t: d(u, t) + (1/2) * d(u*u, x) - 0.5*dd(u, x),
    "boundary_cond": lambda u, x, t: u,
    "boundary_func": lambda x: 1.0 if x > 0.0 else -1.0,
    "init_datum": lambda u, x: x - u
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSc Project, Frederik Kelbel')
    parser.add_argument('--it', nargs="?", type=int, default=1000, help='number of iterations')
    args = parser.parse_args()

    solver = DGMSolver(MODEL_CONFIG_1, VISCOUS_BURGERS_CONFIG)
    losses = list(solver.train(args.it))
    plt.plot(losses)
    plt.show()
    print("Real solution: {}".format(burgers_sol(0.3, 0.1)))
    print("Approx solution: {}".format(solver.u(0.3, 0.1)))

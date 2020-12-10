from operators import d
from DGM import DGMSolver
import argparse

#  Network configurations
MODEL_CONFIG_1 = {
    "batch_size": 1,
    "hidden_dim": 16,
    "learning_rate": 1e-3
}

#  PDE configurations
BURGERS_CONFIG = {
    "x_dim": 1,
    "equation": lambda u, x, t: d(u, t) + u * d(u, x),
    "init_datum": lambda u, x: x - u
}
burgers_sol = lambda x, t: x / (1 + t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSc Project, Frederik Kelbel')
    parser.add_argument('--it', nargs="?", type=int, default=1000, help='number of iterations')
    args = parser.parse_args()

    solver = DGMSolver(MODEL_CONFIG_1, BURGERS_CONFIG)
    solver.train(args.it)
    print("Real solution: {}".format(burgers_sol(2.3, 4.1)))
    print("Approx solution: {}".format(solver.u(2.3, 4.1)))

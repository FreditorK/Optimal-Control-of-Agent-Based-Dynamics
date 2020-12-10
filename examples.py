from operators import d
from DGM import DGMSolver

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
    "init_datum": lambda u, x: (0 if x <= 0 else 1) - u
}


if __name__ == "__main__":
    solver = DGMSolver(MODEL_CONFIG_1, BURGERS_CONFIG)
    solver.train(1000)

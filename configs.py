CONFIG_PARABOLIC_PDES = {
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (2, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "RES",
    "optimiser": "Adam",
    "method": "Galerkin"
}

CONFIG_HBJS = {
    "hidden_dim_u": 8,
    "hidden_dim_J": 32,
    "learning_rate": 5e-3,
    "loss_weights": (1, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "RES",
    "optimiser": "Adam",
    "delay_control": 2,
    "alpha_noise": 0.0
}

CONFIG_FBSDES = {
    "batch_size": 64,
    "num_discretisation_steps": 10,
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (1, 1),
    "lr_decay": 0.99,
    "network_type": "MINI",
    "optimiser": "Adam"
}
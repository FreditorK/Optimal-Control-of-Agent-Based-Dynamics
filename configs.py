CONFIG_PARABOLIC_PDES = {
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (2, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "DGM",
    "optimiser": "Adam",
    "method": "Galerkin"
}

CONFIG_HBJS = {
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (1, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "GRU",
    "optimiser": "Adam",
    "delay_control": 1,
    "alpha_noise": 1e-10
}

CONFIG_FBSDES = {
    "batch_size": 64,
    "num_discretisation_steps": 10,
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (1, 1),
    "lr_decay": 0.99,
    "network_type": "GRU",
    "optimiser": "Adam"
}
CONFIG_PARABOLIC_PDES = {
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (2, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "Res",
    "optimiser": "Adam"
}

CONFIG_HBJS = {
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (1, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "GRU",
    "optimiser": "Adam"
}
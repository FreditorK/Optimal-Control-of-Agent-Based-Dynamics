CONFIG_PARABOLIC_PDES = {
    "batch_size": 128,
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (2, 1),
    "lr_decay": 0.99,
    "sampling_method": "uniform",
    "network_type": "DGM"
}

CONFIG_HBJS = {
    "batch_size": 64,
    "hidden_dim": 64,
    "learning_rate": 5e-3,
    "loss_weights": (2, 1, 1),
    "sampling_method": "uniform"
}
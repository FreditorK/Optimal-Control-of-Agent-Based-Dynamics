from operators import d

burgers_config = {
    "burgers_eq": lambda u, x, t: d(u, t) + u * d(u, x),
    "initial_datum": lambda u, x: (0 if x <= 0 else 1) - u
}

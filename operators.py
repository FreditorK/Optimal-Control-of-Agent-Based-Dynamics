from torch.autograd import grad


def d(u, *vars):
    if vars is ():
        return u
    return d(grad(u, vars[0])[0], vars[1:])

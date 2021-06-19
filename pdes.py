
class PDE:
    """
    x_dim: Dimension of the collective variable vector
    domain_func: Function mapping the sampling space to the domain \ boundary
    boundary_func: Function mapping the sampling space to the boundary
    equation: PDE to solve, function of (u, x, t)
    boundary_cond: Boundary condition, e.g. Dirichlet, Neumann, Robin, etc., function of (u, x, t)
    init_datum: Initial datum, i.e. value of u at time t=0, function of (u, x)
    """

    def __init__(self):
        self.x_dim = None
        self.equation = None
        self.boundary_cond = None  # h(x, t)
        self.init_datum = None  # g(x)
        self.init_func = lambda x: x
        self.domain_func = lambda x: x
        self.boundary_func = lambda x: x


class HBJ:
    """
    cost_function: Cost function for main trajectory, function of (u, x, t), F(u, x, t)
    terminal_cost: Terminal cost at time T of the trajectory, function of (x), G(x)
    differential_operator: Differential operator of the HBJ-equation, function of (J, u, x, t), L
    """

    def __init__(self):
        self.var_dims = []
        self.sampling_funcs = []
        self.cost_function = None
        self.terminal_cost = None
        self.differential_operator = None
        self.control_output = lambda x: x

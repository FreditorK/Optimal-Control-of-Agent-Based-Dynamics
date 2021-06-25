
class PDE:
    """
    var_dim: Dimension of the collective variable vector
    domain_func: Function mapping the sampling space to the domain \ boundary
    boundary_func: Function mapping the sampling space to the boundary
    equation: PDE to solve, function of (u, x, t)
    boundary_cond: Boundary condition, e.g. Dirichlet, Neumann, Robin, etc., function of (u, x, t)
    """

    def __init__(self):
        """
        domain_func: Specifies the functions in composition with the sampler, e.g. We sample per function one batch each
        boundary_func: Same as domain functions but subject to the boundary conditions at the corresponding index
        """
        self.var_dim = None
        self.equation = None
        self.domain_func = []  # f(x)
        self.boundary_cond = []  # h(u, x, t)
        self.boundary_func = []  # f(x)


class HBJ:
    """
    cost_function: Cost function for main trajectory, function of (u, x, t), F(u, x, t)
    differential_operator: Differential operator of the HBJ-equation, function of (J, u, x, t), L
    """

    def __init__(self):
        self.var_dim = None
        self.domain_func = []  # f(x)
        self.boundary_cond_J = []  # h(J, x, t)
        self.boundary_func_J = []  # f(x)
        self.boundary_cond_u = []  # h(u, x, t)
        self.boundary_func_u = []  # f(x)
        self.cost_function = None  # F(u, x, t)
        self.differential_operator = None  # L(J, u, x, t)

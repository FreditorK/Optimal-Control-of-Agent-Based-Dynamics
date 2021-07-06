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
        self.sol_dim = 1
        self.var_dim = None
        self.equation = None
        self.domain_func = []  # f(x)
        self.boundary_cond = []  # h(u, x, t)
        self.boundary_func = []  # f(x), batch_size


class HBJ:
    """
    int: var_dim_J: dimension of dependents of value function, e.g. J(x, t) -> 2, J(x, y, t) -> 3
    list: control_vars: Indices indicating the subset of dependents, e.g. J(x, t) u(t) -> 1, J(x, t) u(x, t) -> 0, 1
    function: cost_function: Cost function for main trajectory, function of (u, x, t), F(u, x, t)
    function: differential_operator: Differential operator of the HBJ-equation, function of (J, u, x, t), L
    """

    def __init__(self):
        self.sol_dim = 1  # dimension of control output
        self.var_dim_J = None
        self.cost_function = None  # F(u, x, t)
        self.differential_operator = None  # L(J, u, x, t)
        self.control_vars = []  # indices of variable dependencies of the control function
        self.domain_func = []  # f(x)
        self.boundary_cond_J = []  # h(J, x, t)
        self.boundary_func_J = []  # f(x)
        self.boundary_cond_u = []  # h(u, x, t)
        self.boundary_func_u = []  # f(x)


class FBSDE:
    def __init__(self):
        self.var_dim = None  # dimension of X
        self.terminal_time = None  # terminal time
        self.H = None  # H(X, t)
        self.sigma = None  # σ(X, t)
        self.C = None  # C(X)
        self.terminal_condition = None  # R(X)
        self.D = None  # Matrix, dimension of control output
        self.Gamma = None  # Γ(X, t)

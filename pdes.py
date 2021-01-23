from abc import ABC, abstractmethod


class PDE(ABC):
    """
    x_dim: Dimension of the collective variable vector
    equation: PDE to solve
    boundary_cond: Boundary condition, e.g. Dirichlet, Neumann, Robin, etc.
    boundary_func: Function mapping the sampling space to the boundary
    init_datum: Initial datum, i.e. value of u at time t=0
    """
    @property
    @abstractmethod
    def x_dim(self):
        pass

    @property
    @abstractmethod
    def equation(self):
        pass

    @property
    @abstractmethod
    def boundary_cond(self):
        pass

    @property
    @abstractmethod
    def boundary_func(self):
        pass

    @property
    @abstractmethod
    def init_datum(self):
        pass


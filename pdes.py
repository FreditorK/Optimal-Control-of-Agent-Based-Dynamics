from abc import ABC, abstractmethod


class PDE(ABC):
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
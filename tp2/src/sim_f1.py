import numpy as np
from src.sim import Simulator


class SimF1(Simulator):
    def __init__(self, n: int, npts: int):
        super().__init__(
            n,
            npts,
            [(-5, 5)]*2,
            0, 30,
            [0.25,1,2,5,10,15],
            'f1',
            np.zeros(n)
        )

    def sim(self, x: np.ndarray):
        """
        Simuler f1
        """
        assert x.shape == (self.n,)
        k = np.arange(1,self.n + 1)
        f = np.sum(k * x**2)
        g = 2 * k * x
        h = np.diag(2*(k+1))
        return f, g, h

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        k = np.arange(1,self.n + 1)
        return np.sum(k * x**2)

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        k = np.arange(1,self.n + 1)
        return 2 * k * x

    def hessian(self, x: np.ndarray):
        assert x.shape == (self.n,)
        k = np.arange(1,self.n + 1)
        return np.diag(2*(k+1))

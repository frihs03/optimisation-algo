import numpy as np
from src.sim import Simulator


class SimF2(Simulator):
    def __init__(self, npts: int):
        super().__init__(
            2,
            npts,
            [(-1.5, 1.5), (-.2, 1.5)],
            0, 200,
            [1,2,5,10,20,30,50,100,200],
            'Rosenbrock',
            np.ones(2)
        )

    def sim(self, x: np.ndarray):
        assert x.shape == (self.n,)
        x1,x2 = x
        f = (1-x1)**2 + 100*(x2-x1**2)**2
        g = np.array([-2*(1-x1) - 400*x1*(x2-x1**2) , 200*(x2-x1**2)])
        return f, g, None

    def primal(self, x: np.ndarray):
        assert x.shape == (self.n,)
        x1,x2 = x 
        return (1-x1)**2 + 100*(x2-x1**2)**2

    def gradient(self, x: np.ndarray):
        assert x.shape == (self.n,)
        x1,x2 = x
        return np.array([-2*(1-x1) - 400*x1*(x2-x1**2) , 200*(x2-x1**2)])

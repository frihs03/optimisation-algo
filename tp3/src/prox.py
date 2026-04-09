import numpy as np
from src.simprox import SimulatorProx
from src.algos import Descent


class Proximal(Descent):
    def __init__(
        self,
        nsteps: int,
        oracle: SimulatorProx,
        start: np.ndarray,
        lr: float=1e-3,
        prec: float=1e-6        
    ):
        print("INIT PROXIMAL FILE:", __file__)
        print("lr INIT =", lr)
        print("type(lr) =", type(lr))
        super().__init__(nsteps, oracle, start)
        self.lr = float=1e-3
        self.prec = prec
        self.last_x = None


    def stop(self, f, g, h, it: int):
        if self.last_x is not None:
            if np.linalg.norm(self.x - self.last_x) < self.prec:
                return True
        return False


    def update(self, f, g, h):
        del f, h
        self.last_x = self.x.copy()
        y = self.x - self.lr * g
        self.x = self.oracle.g_prox(y, self.lr)

        return self.x
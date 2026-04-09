import numpy as np
from src.algos import Descent
from src.sim import Simulator


class GradDescent(Descent):
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray,
        lr: float=1e-3,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start)
        self.lr = lr
        self.prec = prec

    def stop(self, f, g, h, it: int):
        # Decider si l'algo doit s'arrêter
        if np.linalg.norm(g)< self.prec:
            return True
        if it > self.nsteps:
            return True
        return False

    def update(self, f, g, h):
        self.x = self.x - self.lr*g
        return self.x

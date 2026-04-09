import numpy as np
from src.gd import GradDescent
from src.sim import Simulator
from scipy.optimize import line_search


class WolfeLineSearch(GradDescent):
    def __init__(
        self,
        nsteps: int,
        oracle: Simulator,
        start: np.ndarray,
        m1: float=1e-4, m2: float=0.9,
        ls_max=50,
        prec: float=1e-6
    ):
        super().__init__(nsteps, oracle, start)
        self.m1 = m1
        self.m2 = m2
        self.ls_max = ls_max
        self.prec = prec

    def stop(self, f, g, h, it: int):
        # Decider si l'algo doit s'arrêter
        if np.linalg.norm(g)< self.prec:
            return True
        return False

    def update(self, f, g, h):
        # Mettre à jour le vecteur x selon le gradient.
        self.x = self.x - self.lr * g
        return self.x
    
    def l_search(self, dir):
        """
        Fonction de recherche linéaire de Wolfe-Armijo

        Args
        ----
        dir: ndarray
            direction de recherche
        """
        gamma = line_search(
            self.oracle.primal,
            self.oracle.gradient,
            self.x,
            dir,
            gfk=None, old_fval=None, old_old_fval=None, args=(),
            c1=self.m1,
            c2=self.m2,
            amax=self.ls_max
        )[0]
        return gamma

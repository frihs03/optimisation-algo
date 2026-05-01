"""
Functions specific to Gradient Descent
"""

from typing import Optional, Tuple, Callable
import numpy as np


def gd_stepsize_start(
    n: int,
    mu: float,
    L: float
) -> float:
    """
    A function to choose the starting step-size of the algorithm, i.e. its step-size for the first iteration.

    Parameters
    ----------
    n: int
        number of samples of the data distribution
    mu: float
        strong-convexity constant of the objective function
    L: float
        (Approximate ?) Lipschitz constant of the gradient of the objective.
    """
    # We choose step size = 1/L since L is the Lipschitz constant of the gradient.
    # This is the largest safe step size that guarantees convergence for gradient descent,
    return 1.0 /L


def gd_stepsize(it: int, start: float) -> float:
    """
    A function to choose the step-size of the algorithm depending on the current iteration number.

    Parameters
    ----------
    it: int
        current iteration number
    start: float
        first-iteration step-size chosen
    """
    #In Gradient Descent, we use a constant step size (start) since it guarantees the convrgence.
    return start


def gd_step(
    x: np.ndarray,
    n: int,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
) -> Tuple[np.ndarray]:
    """
    This function performs the step of this Deterministic Gradient Descent algorithm.
    Starting at ``x``, it outputs the next state of the algorithm as a 1-uple containing the next state.
    """
    #apply the formula of the x_new
    g = grad(x,None)
    #print(type(g))
    v = x - g * stepsize
    #print(type(v))
    x_new = prox(v,stepsize)
    #print(type(x_new))
    return (x_new,)

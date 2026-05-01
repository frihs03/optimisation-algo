"""
Functions specific to SAGA
TODO (5)
"""

from typing import Optional, Tuple, Callable
import numpy as np


def initialize_gradients_buffer(n: int, d: int) -> np.ndarray:
    return np.zeros((n,d))
    

def saga_stepsize_start(
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
    # same as SGD and GD
    return 1.0/L


def saga_stepsize(it: int, start: float) -> float:
    """
    A function to choose the step-size of the algorithm depending on the current iteration number.

    Parameters
    ----------
    it: int
        current iteration number
    start: float
        first-iteration step-size chosen
    """
    #constant stepsize
    return start 


def saga_step(
    x: np.ndarray,
    n: int,
    grad: Callable[[np.ndarray, Optional[int]], np.ndarray],
    prox: Callable[[np.ndarray, float], np.ndarray],
    stepsize: float,
    gradients_buffer: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function performs the step of this SAGA algorithm.
    Starting at ``x`` with a buffer ``gradients_buffer`` containing n past gradients, it outputs the next state of the algorithm as a 2-uple containing the next state and the updated buffer.
    """
    i = np.random.randint(0,n)
    g_i = grad(x,i)
    ai = gradients_buffer[i]
    a = np.mean(gradients_buffer,axis = 0)
    v = x - stepsize * (g_i -ai +a)
    gradients_buffer[i] = g_i
    x_new = prox(v,stepsize)
    return x_new,gradients_buffer

    

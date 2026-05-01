import numpy as np
from typing import Union
from .solvers import GD, SGD, SAGA
from .regularize import reg_l1, reg_l2


class LogisticRegression:
    """
    Parameters
    ----------
    l1: float
        Used to specify the L1 regression penalty
    l2: float
        Used to specify the L2 regression penalty
    solver: str, {'gd', 'sgd', 'saga'}
        Optimization method
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD,
        one iteration is one epoch i.e, one pass through the data, while for
        SGD, SAG and SAGA, one iteration uses only one data point.

    Attributes
    ----------
    coef_: array
        Coefficient of the features in the decision function.
    """

    SOLVER_SELECTOR = {
        'gd': GD,
        'sgd': SGD,
        'saga': SAGA
    }

    def __init__(self, solver='gd', l1=0.0, l2=0.05, max_iter=100):

        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter

        _solver = LogisticRegression.SOLVER_SELECTOR.get(solver, None)
        if _solver is None:
            raise BaseException("Invalid Solver")
        self.solver = _solver

    def fit(self, A, b):
        r"""
        To fit the model to the data (A,b),
        we minimise the empirical risk over x:

        ..math::
            1/n \sum_{i=1}^n LogisticLoss(b_i, A[i]^T x) + l2/2 * ||x||^2,

        where l2 is the L2 regularization coefficient.
        """

        # Dataset size
        n = len(b)

        # Smoothness constant
        L = 0.25 * max(np.linalg.norm(A, 2, axis=1))**2 + self.l2

        # Strong convexity constant
        mu = self.l2

        # Initialisation of the solver
        x0 = np.zeros(A.shape[1])

        # Compute the grad wrt parameters for current data
        def _grad_for_Ab(
            x,
            i: Union[None, int],
        ) -> np.ndarray:
            return self.grad(x, A, b, i)

        # Last iterate of the optimization, and table of all the iterates
        self.coef_, self._coef_tab = self.solver(
            x0,  # start point
            _grad_for_Ab,  # gradient closure
            self.prox,  # prox operator closure
            self.max_iter,  # max number of iterations
            n,  # number of students in the dataset
            L,  # Lips constant of the gradient of the smooth part of the loss
            mu  # strong convexity constant of the loss
        )

        self._empirical_risk = lambda x: sum([
            sum(self._logistic_loss(A, b, x)) / n,
            reg_l2(x, self.l2),
            reg_l1(x, self.l1)
        ])

    def grad(self, x, A, b, i=None):
        '''
        Gradient of the objective function at x.
        If no i is given, grad should return the (full-)batch gradient.
        Otherwise, it should return the partial gradient associated the the
        datapoint (A[i], b[i]).
        '''

        # Dataset size
        n = len(b)

        if i is None:  # return batch gradient
            output = -b/(1 + np.exp(b*np.dot(A, x)))
            output = output @ A / n
            output += self.l2*x
        else:  # return partial gradient associated the datapoint (A[i], b[i])
            bi = b[i]
            ai = A[i]
            output = (-bi * ai) / (1 + np.exp(1 + bi* np.dot(ai,x))) 
            output += self.l2 *x
            # output is the simple grad of f(xi)
            print(type(output))
        return output

    def prox(self, x, stepsize):
        # Proximal operator for the L1 norm (lambda * ||x||_1).
        # It comes from solving: min_x lambda*||x||_1 + (1/(2*stepsize))||x - v||^2.
        p = np.sign(x) * np.maximum(np.abs(x) - stepsize * self.l1, 0)
        return p

    def decision_function(self, A):
        """
        Predict confidence scores for the samples in A.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return 1.0/(1 + np.exp(-np.dot(A, self.coef_)))

    def predict(self, A):
        """
        Predict class labels for samples in A.
        """
        if not hasattr(self, "coef_"):
            raise BaseException("The model hasn't been fitted yet.")

        return [[-1, 1][p >= 0.5] for p in self.decision_function(A)]

    def score(self, A, b):
        """
        Returns the mean accuracy on the given test data and labels.
        """
        return np.sum(self.predict(A) == b)/len(b)

    # _____________ private functions _____________

    def _logistic_loss(self, A, b, x):
        return np.log(1 + np.exp(-b*np.dot(A, x)))

import warnings
import numpy as np
import matplotlib.pyplot as plt
from src.simprox import SimulatorProx


GRADE_GOOD_SPLIT = 11
EPS = 1e-14


class SimulatorStudentsDataset(SimulatorProx):
    def __init__(
        self,
        title: str,
        filename: str = 'student.npz',
        lam1: float = 0.03,
        lam2: float = 0.1
    ):
        # ==== File reading
        dat_file = np.load(filename)

        # ==== Train part of data
        self.A = dat_file['A_learn']
        final_grades = dat_file['b_learn']
        self.b = self.init_b(final_grades)

        # ==== Test part of data
        self.A_test = dat_file['A_test']
        final_grades_test = dat_file['b_test']
        self.b_test = self.init_b(final_grades_test)

        # ==== Hyper-parameters
        d = 27 # features
        self.n = d+1 # with the intercept

        super().__init__(self.n, title)

        self.lam2 = lam2  # for the 2-norm regularization best:0.1
        self.lam1 = lam1  # for the 1-norm regularization best:0.03
        self.L = self.lipschitz_constant()

    def sim(self, x: np.ndarray):
        return self.F(x), self.f_grad(x), None

    def init_b(self, grades: np.ndarray) -> np.ndarray:
        b = (grades > GRADE_GOOD_SPLIT) * 2.0 - 1.0
        return b

    def lipschitz_constant(self) -> float:
        return 0.25 * np.max(
            np.linalg.norm(
                self.A,
                2,
                axis=1
            )
        ) ** 2 + self.lam2

    def logistic_loss(self, x: np.ndarray) -> float:
        try:
            return self.logistic_loss_vectorized(x)
        except NotImplementedError:
            # Some code that is inneficient to compute the first
            #   part of the loss
            warnings.warn("Logistic loss is not vectorized")
            m = self.A.shape[0]
            l = 0.0
            for i in range(m):
                linear = np.dot(self.A[i], x)
                if self.b[i] > 0 :
                    l += np.log(1 + np.exp(-linear)) 
                else:
                    l += np.log(1 + np.exp(linear)) 
            return l / m

    def logistic_loss_vectorized(self, x: np.ndarray) -> float:
        z = self.A @ x
        bz = self.b * z
        return np.mean(np.logaddexp(0, -bz))
        

    def l2_regularization(self, x: np.ndarray) -> float:
        return self.lam2 / 2.0 * np.dot(x,x)

    def f(self, x: np.ndarray) -> float:
        return self.logistic_loss(x) + self.l2_regularization(x)

    def f_grad(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.f_grad_vectorized(x)
        except NotImplementedError:
            # Some code that is inneficient to compute the gradient
            warnings.warn("Logistic loss grad is not vectorized")
            grad = np.zeros(self.n)
            A = self.A
            m = A.shape[0]
            for i in range(m):
                linear = np.dot(A[i], x)
                if self.b[i] > 0:
                    grad += -A[i] / (1 + np.exp(linear)) 
                else:
                    grad += A[i] / (1 + np.exp(-linear)) 
            return grad/m + self.lam2 * x

    def f_grad_vectorized(self, x: np.ndarray) -> np.ndarray:
        z = self.A @ x
        bz = self.b * z
        q = 1 / (1+ np.exp(bz))
        grad = -self.A.T @ (self.b * q) / self.A.shape[0]
        return grad + self.lam2 * x

        
    def l1_regularization(self, x: np.ndarray) -> float:
        return self.lam1 * np.linalg.norm(x, 1)

    def g(self, x: np.ndarray) -> float:
        return self.l1_regularization(x)

    def g_prox(self, y: np.ndarray, gamma: float) -> np.ndarray:
        #resultat explicite de g_prox
        a = gamma * self.lam1
        return np.sign(y) * np.maximum(np.abs(y) - a, 0)

    def prox(self, y: np.ndarray, lr: float) -> np.ndarray:
        return self.g_prox(y, lr)

    def prediction_train(self, x: np.ndarray, do_print: bool):
        try:
            return self.prediction_train_vectorized(x, do_print)
        except NotImplementedError:
            # Some code that is inneficient to compute the gradient
            warnings.warn("Logistic loss grad is not vectorized")
            A = self.A
            m = A.shape[0]
            pred = np.zeros(m)
            perf = 0
            for i in range(m):
                p = 1.0 / (1 + np.exp(-np.dot(A[i], x)))
                if p > 0.5:
                    pred[i] = 1.0
                    if self.b[i] > 0:
                        correct = "True"
                        perf += 1
                    else:
                        correct = "False"
                    if do_print:
                        print(
                            "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                                int(self.b[i]),
                                1,
                                (p - 0.5)*200,
                                correct
                            )
                        )
                else:
                    pred[i] = -1.0
                    if self.b[i] < 0:
                        correct = "True"
                        perf += 1
                    else:
                        correct = "False"
                    if do_print:
                        print(
                            "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                                int(self.b[i]),
                                -1,
                                100 - (0.5 - p)*200,
                                correct
                            )
                        )
            return pred, float(perf) / m

    def prediction_train_vectorized(self, x: np.ndarray, do_print: bool):
        m = self.A.shape[0]
        z = self.A @ x
        p = 1.0 / (1+np.exp(-z))
        #predictions
        pred = np.where(p>0.5 ,1 ,-1)
        #performences
        perf = np.mean(pred == self.b)
        #printing
        if do_print:
            for i in range(m):
                if pred[i] == 1:
                    confidence = (p[i] - 0.5) * 200
                else:
                    confidence = 100 - (0.5 - p[i]) * 200

                correct = "True" if pred[i] == self.b[i] else "False"

                print(
                    "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                        int(self.b[i]),
                        int(pred[i]),
                        confidence,
                        correct
                    )
                )

        return pred, perf

    def prediction_test(self, x: np.ndarray, do_print: bool):
        try:
            return self.prediction_test_vectorized(x, do_print)
        except NotImplementedError:
            A_test = self.A_test
            m_test = A_test.shape[0]
            pred = np.zeros(m_test)
            perf = 0
            for i in range(m_test):
                p = 1.0 / (1 + np.exp(-np.dot(A_test[i], x )))
                if p > 0.5:
                    pred[i] = 1.0
                    if self.b_test[i] > 0:
                        correct = "True"
                        perf += 1
                    else:
                        correct = "False"
                    if do_print:
                        print(
                            "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                                int(self.b[i]),
                                1,
                                (p - 0.5)*200,
                                correct
                            )
                        )
                else:
                    pred[i] = -1.0
                    if self.b_test[i] < 0:
                        correct = "True"
                        perf += 1
                    else:
                        correct = "False"
                    if do_print:
                        print(
                            "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                                int(self.b[i]),
                                -1,
                                100 - (0.5 - p)*200,
                                correct
                            )
                        )
            return pred, float(perf) / m_test

    def prediction_test_vectorized(self, x: np.ndarray, do_print: bool):
        m = self.A_test.shape[0]
        z = self.A_test @ x
        p = 1.0 / (1-np.exp(-z))
        #predictions
        pred = np.where(p>0.5 ,1 ,-1)
        #performences
        perf = np.mean(pred == self.b_test)
        #printing
        if do_print:
            for i in range(m):
                if pred[i] == 1:
                    confidence = (p[i] - 0.5) * 200
                else:
                    confidence = 100 - (0.5 - p[i]) * 200

                correct = "True" if pred[i] == self.b_test[i] else "False"

                print(
                    "True class: {:d} \t-- Predicted: {} \t(confidence: {:.1f}%)\t{}".format(
                        int(self.b_test[i]),
                        int(pred[i]),
                        confidence,
                        correct
                    )
                )

        return pred, perf

    # === Plotting methods

    def plot_loss(self, x_tab: np.ndarray):
        plt.figure()
        plt.plot(
            list(map(self.F, (x_tab[i, :] for i in range(x_tab.shape[0])))),
            color="black",
            linewidth=1.0,
            linestyle="-"
        )
        plt.title(self.title)
        plt.grid(True)
        plt.show()

    def plot_support(self, x_tab: np.ndarray, period: int = 40):
        plt.figure()

        abs, ord = np.where(np.abs(x_tab)[::period] > EPS)
        plt.plot(period * abs, ord, 'ko')

        plt.grid(True)
        plt.title(self.title)
        plt.ylabel('Non-null Coordinates')
        plt.xlabel('Nb. Iterations')
        plt.ylim(-1, self.n)
        plt.yticks(np.arange(0, self.n))
        plt.show()

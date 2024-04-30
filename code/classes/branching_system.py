import numpy as np
from non_linear_system import NonLinearSystem


class BranchingSystem(NonLinearSystem):
    """
    This class represents a branching system of non-linear equations,
    which is a non-linear system of equations itself. Rootfinding method
    can be applied to this system to find its solutions: bifurcation points.
    """

    # TODO: see section 5.5 (Seydel) on how to determine l and k such that the
    # approximation of the initial h is good
    l_default = 1
    k_default = 1

    def __init__(self, nls: NonLinearSystem, parameterName: str) -> None:
        """
        Initialize the branching system.

        Parameters
        nls : NonLinearSystem; the non-linear system of equations
        parameterName : str; the name of parameter attribte of nls to be continued
        """
        self.nls = nls
        self.n = nls.evaluate().shape[0]
        self.parameterName = parameterName
        self._l = self.l_default
        self._k = self.k_default
        self.h = self.determine_initial_h()

    # main methods
    def evaluate(self) -> np.ndarray:
        f = self.nls.evaluate()
        df = self.nls.evaluate_derivative()
        F = np.concatenate((f, df @ self.h, [self.h[self.k] - 1]))
        return F

    def evaluate_derivative(self) -> np.ndarray:
        raise NotImplementedError(
            "Method evaluate_derivative not implemented in BranchingSystem"
        )

    def evaluate_derivative_finite_difference(self, step: float = 1e-6) -> np.ndarray:
        F = self.evaluate()
        currentSolution = self.get_current_solution()
        # dF = np.zeros((2 * self.n + 1, 2 * self.n + 1))
        F_step = np.zeros((2 * self.n + 1, 2 * self.n + 1))
        for i in range(2 * self.n + 1):
            Y = currentSolution.copy()
            Y[i] += step
            self.set_current_solution(Y)
            F_step[i, :] = self.evaluate()
            self.set_current_solution(currentSolution)
        dF = (F_step - F) / step
        return dF.T

    def get_current_solution(self) -> np.ndarray:
        parameter = getattr(self.nls, self.parameterName)
        Y = np.concatenate((self.nls.evaluate(), [parameter], self.h))
        return Y

    def set_current_solution(self, solution: np.ndarray) -> None:
        y = solution[: self.n]
        parameter = solution[self.n]
        h = solution[self.n + 1 :]
        self.nls.set_current_solution(y)
        setattr(self.nls, self.parameterName, parameter)
        self.h = h

    def update_solution(self, update) -> None:
        y_update = update[: self.n]
        parameter_update = update[self.n]
        h_update = update[self.n + 1 :]
        self.nls.update_solution(y_update)
        parameter = getattr(self.nls, self.parameterName)
        setattr(self.nls, self.parameterName, parameter + parameter_update)
        self.h += h_update

    # determine initial approximation of h
    def determine_initial_h(self) -> np.ndarray:
        J = self.nls.evaluate_derivative()
        e_l = np.zeros(self.n)
        e_l[self.l] = 1
        e_k = np.zeros(self.n)
        e_k[self.k] = 1
        J_lk = J
        J_lk[self.l, :] = e_k
        return np.linalg.solve(J_lk, e_l)

    # getters
    @property
    def l(self) -> int:
        return self._l

    @property
    def k(self) -> int:
        return self._k

    # setters
    @l.setter
    def l(self, l: int) -> None:
        self._l = l

    @k.setter
    def k(self, k: int) -> None:
        self._k = k

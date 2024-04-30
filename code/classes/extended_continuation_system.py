import numpy as np
from non_linear_system import NonLinearSystem


class ExtendedSystem(NonLinearSystem):
    """
    Non-Linear system used for corrector steps in continuation
    """

    def __init__(
        self,
        nls: NonLinearSystem,
        previousSolution: np.ndarray,
        parameterName: str,
        stepsize: float,
        tuneFactor: float,
    ) -> None:
        """
        To be initialized directly after any kind of predictor step
        """
        self.nls = nls
        self.n = nls.evaluate().shape[0]
        self.previousY = previousSolution[: self.n]
        self.previousParameter = previousSolution[self.n]
        self.parameterName = parameterName
        self.stepsize = stepsize
        self.tuneFactor = tuneFactor

    # main methods
    def evaluate(self) -> np.ndarray:
        currentSolution = self.get_current_solution()
        currentY = currentSolution[: self.n]
        currentParameter = currentSolution[self.n]
        f = self.nls.evaluate()
        p = (
            self.tuneFactor * np.sum((currentY - self.previousY) ** 2)
            + (1 - self.tuneFactor) * (currentParameter - self.previousParameter) ** 2
            - self.stepsize**2
        )
        return np.append(f, p)

    def evaluate_derivative(self) -> np.ndarray:
        raise NotImplementedError(
            "Method evaluate_derivative not implemented in BranchingSystem"
        )

    def evaluate_derivative_finite_difference(self, step: float = 1e-6) -> np.ndarray:
        F = self.evaluate()
        currentSolution = self.get_current_solution()
        F_step = np.zeros((self.n + 1, self.n + 1))
        for i in range(self.n + 1):
            Y = currentSolution.copy()
            Y[i] += step
            self.set_current_solution(Y)
            F_step[i, :] = self.evaluate()
            self.set_current_solution(currentSolution)
        dF = (F_step - F) / step
        return dF.T
        # df_dsol = self.nls.evaluate_derivative()
        # df_dparam = self.derivativeParam()
        # currentY = self.nls.get_current_solution()
        # currentParameter = getattr(self.nls, self.parameterName)
        # dsol_ds = (currentY - self.previousY) / self.stepsize
        # dparam_ds = (currentParameter - self.previousParameter) / self.stepsize
        # dp_dsol = 2 * self.tuneFactor * dsol_ds
        # dp_dparam = 2 * (1 - self.tuneFactor) * dparam_ds
        # dF = np.vstack(
        #     (
        #         np.hstack((df_dsol, df_dparam[:, np.newaxis])),
        #         np.hstack((dp_dsol, dp_dparam)),
        #     )
        # )
        return dF

    def get_current_solution(self) -> np.ndarray:
        currentY = self.nls.get_current_solution()
        currentParameter = getattr(self.nls, self.parameterName)
        return np.append(currentY, currentParameter)

    def set_current_solution(self, solution: np.ndarray) -> None:
        newY = solution[: self.n]
        newParameter = solution[self.n]
        self.nls.set_current_solution(newY)
        setattr(self.nls, self.parameterName, newParameter)

    def update_solution(self, update) -> None:
        updateY = update[: self.n]
        updateParameter = update[self.n]
        self.nls.update_solution(updateY)
        parameter = getattr(self.nls, self.parameterName)
        setattr(self.nls, self.parameterName, parameter + updateParameter)

    # getters

    # setters

    # helper
    def derivativeParam(self, h: float = 1e-5) -> np.ndarray:
        """
        Calculate the derivative of the non-linear system w.r.t. the parameter.
        """
        f = self.nls.evaluate()
        setattr(self.nls, self.parameterName, self.previousParameter + h)
        dF_param = (self.nls.evaluate() - f) / h

        # reset parameter
        setattr(self.nls, self.parameterName, self.previousParameter)
        return dF_param

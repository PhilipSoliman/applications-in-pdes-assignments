import numpy as np
from tqdm import tqdm
from non_linear_system import NonLinearSystem
from root_finding import RootFinding


class Continuation(NonLinearSystem):
    """
    Performs on continuation on a given NonLinearSystem object
    assuming the current solution is a solution.

    If a new solution is found, the NonLinearSystem object is updated.
    Avaliable methods are:
    - arclength
    """

    def __init__(self, maxiter: int) -> None:
        self.maxiter = maxiter
        self.convergence = False
        self.setDefaultAttributes()

    def setDefaultAttributes(self) -> None:
        self.parameterName = ""
        self.stepsize = 0.0
        self.tune_factor = 1.0
        self.tolerance = 1e-5

    # method handles
    def arclength(
        self,
        nls: NonLinearSystem,
        parameterName: str,
        stepsize: float,
        tune_factor: float = 1.0,
        tolerance: float = 1e-5,
    ) -> np.ndarray:
        """
        Perform continuation using the arclength method. Calls main method
        using the specified NonLinearSystem and parameter.
        """
        if not hasattr(nls, parameterName):
            raise ValueError(
                f"Parameter attribute {parameterName} not found in given NonLinearSystem."
            )
        self.parameterName = parameterName
        self.method = "ARC"
        self.tune_factor = tune_factor
        self.stepsize = stepsize
        self.tolerance = tolerance
        return self.execute(nls)

    # main method
    def execute(self, nls: NonLinearSystem) -> None:
        self.converged = False

        if self.method == "ARC":
            self.arclengthAlgorithm(nls, self.stepsize, self.tolerance)
        else:
            raise ValueError("Method not specified and/or implemented.")

        if not self.convergence:
            print(
                f"'{self.method}' continuation did not converge within the maximum number of iterations. Exiting..."
            )

        self.setDefaultAttributes()

    # arclength method
    def arclengthAlgorithm(
        self, nls: NonLinearSystem, stepsize, tolerance: float = 1e-5, h: float = 1e-4
    ) -> None:
        solution = nls.get_current_solution()
        parameter = getattr(nls, self.parameterName)

        # predictor step
        dF_param = self.derivativeParam(nls, h)
        dF_sol = nls.evaluate_derivative()

        dsol_dparam = -np.linalg.solve(dF_sol, dF_param)
        dparam_ds = 1 / np.sqrt(1 + self.tune_factor * np.sum(dsol_dparam**2))
        dsol_ds = dsol_dparam * dparam_ds

        predictorStepSolution = stepsize * dsol_ds
        predictorStepParameter = stepsize * dparam_ds

        predictorSolution = solution + predictorStepSolution
        predictorParameter = parameter + predictorStepParameter

        # update non-linear system
        nls.update_solution(predictorStepSolution)
        setattr(nls, self.parameterName, predictorStepParameter)

        # corrector iterations
        self.convergence = False
        error = 2 * tolerance
        i = 0
        while error > tolerance and i < self.maxiter:
            # set initial values of the corrector
            if i == 0:
                correctorSolution = predictorSolution
                correctorParam = predictorParameter

            # evaluate non-linear system and its derivative both w.r.t. the parameter and the solution
            dF_param = self.derivativeParam(nls, h)
            dF_sol = nls.evaluate_derivative()

            # intermediate variables
            z1 = np.linalg.solve(dF_sol, -self.F)
            z2 = np.linalg.solve(dF_sol, dF_param)

            # parametrisation p and its derivatives
            p = (
                self.tuning_factor * np.sum((correctorSolution - solution) ** 2)
                + (1 - self.tuning_factor) * (correctorParam - parameter) ** 2
                - stepsize**2
            )
            dp_dsol = 2 * self.tune_factor * (correctorSolution - solution)[:, None]
            dp_dparam = 2 * (1 - self.tune_factor) * (correctorParam - parameter)

            # corrector step
            correctorStepParameter = (-p - dp_dsol.T @ z1) / (
                dp_dparam - dp_dsol.T @ z2
            )
            correctorStepSolution = z1 + z2 * correctorStepParameter

            correctorSolution += correctorStepSolution
            correctorParam += correctorStepSolution

            # update non linear system
            nls.update_solution(correctorSolution)
            setattr(nls, self.parameterName, correctorParam)

            # update error
            correctorStep = np.concatenate(
                (correctorStepSolution, correctorStepParameter)
            )
            error = np.linalg.norm(correctorStep)

            i += 1

        if error <= tolerance:
            self.convergence = True
        else:
            self.convergence = False

            # restore non-linear system
            nls.set_current_solution(solution)
            setattr(nls, self.parameterName, parameter)

    def derivativeParam(
        self, nls: NonLinearSystem, parameter: float, h: float = 1e-6
    ) -> np.ndarray:
        """
        Calculate the derivative of the non-linear system w.r.t. the parameter.
        """
        self.F = nls.evaluate()
        setattr(nls, self.parameterName, parameter + h)
        dF_param = (nls.evaluate() - self.F) / h

        # reset parameter
        setattr(nls, self.parameterName, parameter)
        return dF_param

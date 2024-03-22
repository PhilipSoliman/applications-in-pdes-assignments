import numpy as np
from tqdm import tqdm
from non_linear_system import NonLinearSystem
from root_finding import RootFinding


class Continuation:
    """
    Performs on continuation on a given NonLinearSystem object
    assuming the current solution is a solution.

    If a new solution is found, the NonLinearSystem object is updated.
    Avaliable methods are:
    - arclength (ARC)
    """

    def __init__(self, tolerance=1e-5, maxiter: int = 100) -> None:
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.maxRetries = 3
        self.convergence = False
        self.setDefaultAttributes()
        self.output = False

    def setDefaultAttributes(self) -> None:
        self.parameterName = ""
        self.stepsize = 0.0
        self.tune_factor = 0.0
        self.stableBranch = None
        self.remainingRetries = 10

    # method handles
    def arclength(
        self,
        nls: NonLinearSystem,
        parameterName: str,
        stepsize: float,
        tune_factor: float = 1.0,
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
        return self.execute(nls, stepsize)

    # main method
    def execute(self, nls: NonLinearSystem, stepsize: float) -> None:
        self.convergence = False

        if self.method == "ARC":
            errors = self.arclengthAlgorithm(nls, stepsize, self.tolerance)
        else:
            raise ValueError("Method not specified and/or implemented.")

        if self.convergence:
            self.stableBranch = Continuation.checkStability(nls)
            self.print(f"{self.method} continuation converged.")
        else:
            self.remainingRetries -= 1
            if self.remainingRetries > 0:  # refinement
                self.print(
                    f"{self.method} continuation did not converge within the maximum number of iterations. Retrying with smaller stepsize :{self.stepsize:.2e}..."
                )
                self.stepsize /= 10
                self.execute(nls, self.stepsize)
            else:  # restart by finding a new root
                self.print(
                    f"{self.method} continuation did not converge within the maximum number of retries. finding new root..."
                )
                rootfinding = RootFinding(self.tolerance, self.maxiter)
                rootfinding.output = self.output
                rootfinding.newtonRaphson(nls)

                # reset stepsize & retries
                self.stepsize *= 10 ** (self.maxRetries - self.remainingRetries)
                self.remainingRetries = 10

                self.execute(nls, self.stepsize)

        self.setDefaultAttributes()

        return errors

    # arclength method
    def arclengthAlgorithm(
        self, nls: NonLinearSystem, stepsize, tolerance: float = 1e-5, h: float = 1e-4
    ) -> list:
        solution = nls.get_current_solution()
        parameter = getattr(nls, self.parameterName)

        # predictor step
        dF_param = self.derivativeParam(nls, parameter, h)
        dF_sol = nls.evaluate_derivative()

        dsol_dparam = -np.linalg.solve(dF_sol, dF_param)
        dparam_ds = 1 / np.sqrt(1 + self.tune_factor * np.sum(dsol_dparam**2))
        dsol_ds = dsol_dparam * dparam_ds

        predictorSolution = solution + stepsize * dsol_ds
        predictorParameter = parameter + stepsize * dparam_ds

        # update non-linear system
        nls.set_current_solution(predictorSolution)
        setattr(nls, self.parameterName, predictorParameter)

        # corrector iterations
        self.convergence = False
        error = 2 * tolerance
        i = 0
        errors = []
        while error > tolerance and i < self.maxiter:
            # set initial values of the corrector
            if i == 0:
                correctorSolution = predictorSolution
                correctorParameter = predictorParameter

            # evaluate non-linear system and its derivative both w.r.t. the parameter and the solution
            dF_param = self.derivativeParam(nls, parameter, h)
            dF_sol = nls.evaluate_derivative()

            # intermediate variables
            # z1 = np.linalg.solve(dF_sol, -self.F)
            # z2 = np.linalg.solve(dF_sol, dF_param)

            # parametrisation p and its derivatives
            dsol_ds = (correctorSolution - solution) / stepsize
            dparam_ds = (correctorParameter - parameter) / stepsize
            p = (
                self.tune_factor * np.sum((correctorSolution - solution) * dsol_ds)
                + (1 - self.tune_factor) * (correctorParameter - parameter) * dparam_ds
                - stepsize**2
            )
            dp_dsol = 2 * self.tune_factor * dsol_ds
            dp_dparam = 2 * (1 - self.tune_factor) * dparam_ds

            # extended system
            F_ext = np.append(self.F, p)
            dF_ext = np.vstack(
                (
                    np.hstack((dF_sol, dF_param[:, np.newaxis])),
                    np.hstack((dp_dsol, dp_dparam)),
                )
            )

            # corrector step (basically a Newton-Raphson step on extended system)
            correctorStep = -np.linalg.solve(dF_ext, F_ext)
            correctorStepSolution = correctorStep[:-1]
            correctorStepParameter = correctorStep[-1]

            correctorSolution += correctorStepSolution
            correctorParameter += correctorStepParameter

            # update non linear system
            nls.set_current_solution(correctorSolution)
            setattr(nls, self.parameterName, correctorParameter)

            # update error
            error = np.linalg.norm(correctorStep)
            errors.append(error)

            i += 1

        if error <= tolerance:
            self.convergence = True
        else:  # reset to previous solution
            nls.set_current_solution(solution)
            setattr(nls, self.parameterName, parameter)

        return errors

    def derivativeParam(
        self, nls: NonLinearSystem, parameter: float, h: float
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

    @staticmethod
    def checkStability(nls: NonLinearSystem) -> None:
        """
        Check the stability of the current solution.
        """
        jacobian = nls.evaluate_derivative()
        eigvals = np.linalg.eigvals(jacobian)
        return np.all(np.real(eigvals) < 0)

    def print(self, msg: str, **kwargs) -> None:
        if self.output:
            print(msg, **kwargs)

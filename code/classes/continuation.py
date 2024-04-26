import numpy as np
from non_linear_system import NonLinearSystem
from tqdm import tqdm


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
        self.continuedSolution = None
        self.previousSolution = None

    def setDefaultAttributes(self) -> None:
        self.parameterName = ""
        self.stepsize = 0.0
        self.tuneFactor = 0.0
        self.stableBranch = None
        self.remainingRetries = 10

    # method handles
    def arclength(
        self,
        nls: NonLinearSystem,
        parameterName: str,
        stepsize: float,
        tuneFactor: float = 1.0,
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
        self.tuneFactor = tuneFactor
        self.stepsize = stepsize
        return self.continuation(nls, stepsize)

    def continuation(self, nls: NonLinearSystem, stepsize: float) -> None:
        """
        performs single continuation step using the specified method.
        """
        self.convergence = False
        parameter = getattr(nls, self.parameterName)
        self.previousSolution = np.append(nls.get_current_solution(), parameter)

        if self.method == "ARC":
            errors = self.arclengthAlgorithm(nls, stepsize, self.tolerance)
        else:
            raise ValueError("Method not specified and/or implemented.")

        if self.convergence:
            self.stableBranch = self.checkStability(nls)
            self.print(f"{self.method} continuation converged.")
        else:
            self.remainingRetries -= 1
            if self.remainingRetries > 0:  # refinement
                self.stepsize /= 10
                self.print(
                    f"{self.method} continuation did not converge within the maximum number of iterations. Retrying with smaller stepsize :{self.stepsize:.2e}..."
                )
                self.continuation(nls, self.stepsize)
            else:
                self.print(
                    f"{self.method} continuation did not converge within the maximum number of retries. Exiting..."
                )

        return errors

    # automatic continuation loop methods
    def arclengthLoop(
        self,
        nls: NonLinearSystem,
        parameterName: str,
        stepsize: float,
        tuneFactor: float,
        parameterRange: tuple,
        maxContinuations: int,
    ) -> None:
        if not hasattr(nls, parameterName):
            raise ValueError(
                f"Parameter attribute {parameterName} not found in given NonLinearSystem."
            )
        self.parameterName = parameterName
        self.method = "ARC"
        self.tuneFactor = tuneFactor
        self.stepsize = stepsize
        self.parameterRange = parameterRange
        self.maxContinuations = maxContinuations
        return self.continuationLoop(nls)

    def continuationLoop(self, nls: NonLinearSystem) -> tuple[np.ndarray]:
        """
        Perform a continuation loop on the given NonLinearSystem object.
        Aim is to find all branches of solutions in the given parameter range.

        Idea:
        - perform continuation for the given initial solution
        - check if lowest (highest) parameter value still in range
        - if not, exit
        - if yes, check if solution is stable
        - if yes, continue
        - if no, find new root and perform continuation (restart using recursive call)
            - stop when encountering already found branch
        - repeat until maxContinuations reached
        """
        # set initial parameter value
        setattr(nls, self.parameterName, self.parameterRange[0])
        i = 0
        solutionAverages = []
        parameterValues = []
        stableBranch = []
        while True:
            if self.method == "ARC":
                self.arclength(nls, self.parameterName, self.stepsize, self.tuneFactor)
            else:
                raise ValueError("Method not specified and/or implemented.")
            solution = nls.get_current_solution()
            average = np.mean(solution)
            minimum = np.min(solution)
            maximum = np.max(solution)
            solutionAverages.append(average)
            parameter = getattr(nls, self.parameterName)
            parameterValues.append(parameter)
            stableBranch.append(self.stableBranch)  # calculates eigenvalues

            i += 1

            self.print(
                f"Continuation step {i}: {self.parameterName} = {parameter}, avg = {average:.2e}, min = {minimum:.2e}, max = {maximum:.2e}, stable = {self.stableBranch}"
            )

            if not self.convergence:
                self.print(f"Continuation did not converge after {i} steps. Exiting...")
                break

            if i == self.maxContinuations:
                self.print(
                    f"Maximum number of continuations reached ({self.maxContinuations}). Exiting..."
                )
                break

            if parameter > self.parameterRange[1]:
                self.print(
                    f"Parameter range reached end of range ({self.parameterName} >= {self.parameterRange[1]}). Exiting..."
                )
                break

            if self.checkFold(nls):
                self.print("Fold detected. Exiting...")
                break

            # if (
            #     np.linalg.norm(
            #         (self.previousSolution - self.continuedSolution)
            #         / self.previousSolution
            #     )
            #     < 1e-2 and self.previousSolution is not None
            # ):
            #     print("Encountered old branch. Exiting...")
            #     break

        return np.array(solutionAverages), np.array(parameterValues), np.array(stableBranch)

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
        dparam_ds = 1 / np.sqrt(1 + self.tuneFactor * np.sum(dsol_dparam**2))
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

            # parametrisation p and its derivatives
            dsol_ds = (correctorSolution - solution) / stepsize
            dparam_ds = (correctorParameter - parameter) / stepsize
            p = (
                self.tuneFactor * np.sum((correctorSolution - solution) * dsol_ds)
                + (1 - self.tuneFactor) * (correctorParameter - parameter) * dparam_ds
                - stepsize
            )
            dp_dsol = 2 * self.tuneFactor * dsol_ds
            dp_dparam = 2 * (1 - self.tuneFactor) * dparam_ds

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
            self.continuedSolution = np.append(correctorSolution, correctorParameter)
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

    def checkStability(self, nls: NonLinearSystem) -> None:
        """
        Check the stability of the current solution.
        """
        jacobian = nls.evaluate_derivative()
        self.eigvals = np.linalg.eigvals(jacobian)
        return np.all(np.real(self.eigvals) < 0)

    @staticmethod
    def checkFold(nls: NonLinearSystem) -> None:
        """
        Check if a fold bifurcation is encountered.
        """
        jacobian = nls.evaluate_derivative()
        n = jacobian.shape[0]
        if np.linalg.matrix_rank(jacobian) < n:
            return True
        else:
            return False

    def print(self, msg: str, **kwargs) -> None:
        if self.output:
            print(msg, **kwargs)

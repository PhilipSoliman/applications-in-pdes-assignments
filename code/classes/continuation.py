import numpy as np
from branching_system import BranchingSystem
from non_linear_system import NonLinearSystem
from root_finding import RootFinding
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
        self.currentSolution = None

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
        self.currentSolution = np.append(nls.get_current_solution(), parameter)

        if self.method == "ARC":
            errors = self.arclengthAlgorithm2(nls, stepsize, self.tolerance)
        else:
            raise ValueError("Method not specified and/or implemented.")

        if self.convergence:
            self.stableBranch = self.checkStability(nls)
            self.print(
                f"{self.method} continuation converged with stepsize {self.stepsize}."
            )
            self.stepsize *= 10 ** (10 - self.remainingRetries)  # reset stepsize
            self.remainingRetries = 10
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

    def continuationLoop(self, nls: NonLinearSystem) -> dict[np.ndarray]:
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
        solutionMinima = []
        solutionMaxima = []
        parameterValues = []
        stableBranch = []
        bifurcationPoints = []
        self.stableBranch = self.checkStability(nls)
        while True:
            solution = nls.get_current_solution()
            average = np.mean(solution)
            solutionAverages.append(average)
            minimum = np.min(solution)
            solutionMinima.append(minimum)
            maximum = np.max(solution)
            solutionMaxima.append(maximum)
            parameter = getattr(nls, self.parameterName)
            parameterValues.append(parameter)
            stableBranch.append(self.stableBranch)  # calculates eigenvalues

            old_stability = self.stableBranch
            if self.method == "ARC":
                self.arclength(nls, self.parameterName, self.stepsize, self.tuneFactor)
            else:
                raise ValueError("Method not specified and/or implemented.")
            new_stability = self.stableBranch
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

            if old_stability != new_stability:
                self.print("Encountered bifurcation! Determining exact location")
                bifurcationPoint = self.findBifurcation(
                    nls
                )  # automatically updates nls to bifurcation point
                bifurcationPoints.append(bifurcationPoint)
                # break  # TODO: implement restart + branch switching

        solutions = dict(
            average=np.array(solutionAverages),
            minimum=np.array(solutionMinima),
            maximum=np.array(solutionMaxima),
            parameter=np.array(parameterValues),
            stable=np.array(stableBranch),
            bifurcations=bifurcationPoints,
        )
        return solutions

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

    def arclengthAlgorithm2(
        self, nls: NonLinearSystem, stepsize, tolerance: float = 1e-5, h: float = 1e-4
    ) -> list:
        solution = self.currentSolution[:-1]
        parameter = self.currentSolution[-1]

        predictorStep = self.predictorStep(nls)

        predictorSolution = solution + stepsize * predictorStep[:-1]
        predictorParameter = parameter + stepsize * predictorStep[-1]

        # update non-linear system
        nls.set_current_solution(predictorSolution)
        setattr(nls, self.parameterName, predictorParameter)

        # set initial values of the corrector
        correctorSolution = predictorSolution
        correctorParameter = predictorParameter

        # corrector iterations
        self.convergence = False
        error = 2 * tolerance
        i = 0
        errors = []
        while error > tolerance and i < self.maxiter:
            # TODO: make extend system into nls object and use rootfinding
            # extended system
            F = nls.evaluate()
            p = (
                self.tuneFactor * np.sum((correctorSolution - solution) ** 2)
                + (1 - self.tuneFactor) * (correctorParameter - parameter) ** 2
                - stepsize**2
            )
            F_ext = np.append(F, p)

            if i == 0:  # fixed point iteration
                correctorStep = F_ext

            if i >= 0:  # Newton-Raphson step on extended system
                df_dsol = nls.evaluate_derivative()
                df_dparam = self.derivativeParam(nls, parameter, h)
                dsol_ds = (correctorSolution - solution) / stepsize
                dparam_ds = (correctorParameter - parameter) / stepsize
                dp_dsol = 2 * self.tuneFactor * dsol_ds
                dp_dparam = 2 * (1 - self.tuneFactor) * dparam_ds
                dF_ext = np.vstack(
                    (
                        np.hstack((df_dsol, df_dparam[:, np.newaxis])),
                        np.hstack((dp_dsol, dp_dparam)),
                    )
                )
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

    def predictorStep(self, nls) -> np.ndarray:
        """
        Calculate tangent vector to branch for predictor step.
        """
        df_dsol = nls.evaluate_derivative()
        parameter = self.currentSolution[-1]
        df_dparam = self.derivativeParam(nls, parameter, 1e-4)
        n = df_dsol.shape[0]
        k = 0
        tangentSystem = np.zeros((n + 1, n + 1))
        while k < n:
            e_k = np.zeros(n+1)
            e_k[k] = 1
            tangentSystem = np.vstack(
                (np.hstack((df_dsol, df_dparam[:, np.newaxis])), [e_k])
            )
            if np.linalg.matrix_rank(tangentSystem) == n + 1:
                break
            k += 1

        tangent = np.linalg.solve(tangentSystem, e_k)

        return tangent

    # bifurcation detection
    def findBifurcation(self, nls: NonLinearSystem) -> dict:
        """
        Find the exact location of a bifurcation point using a direct method
        see Seydel section 5.4.1 (algorithm 5.4). Only works for bifurcations
        that result in a singular jacobian at the bifurcation point.

        Y = (solution, parameter, h)^T
        F(Y) := branching system
        """
        current_solution = nls.get_current_solution()
        current_parameter = getattr(nls, self.parameterName)
        self.print("Finding bifurcation point...")
        branching_system = BranchingSystem(nls, self.parameterName)
        Y = branching_system.get_current_solution()
        F = branching_system.evaluate()
        rootfinder = RootFinding()
        rootfinder.output = True
        _ = rootfinder.newtonRaphson(branching_system, exact=False)
        if not rootfinder.converged:
            self.print(
                "Bifurcation detection failed. Resetting old initial solution..."
            )
            nls.set_current_solution(current_solution)
            setattr(nls, self.parameterName, current_parameter)
            return
        dF = nls.evaluate_derivative()
        eigvals = np.linalg.eigvals(dF)
        return dict(
            solution=nls.get_current_solution(),
            parameter=getattr(nls, self.parameterName),
            eigvals=eigvals,
            type="...",  # TODO: add type of bifurcation detection
        )

    # helper methods
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

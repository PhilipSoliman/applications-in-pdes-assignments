from pprint import pprint

import numpy as np
from bcolors import bcolors
from branching_system import BranchingSystem
from extended_continuation_system import ExtendedSystem
from non_linear_system import NonLinearSystem
from root_finding import RootFinding
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import fsolve
from tqdm import tqdm


class Continuation:
    """
    Performs on continuation on a given NonLinearSystem object
    assuming the current solution is a solution.

    If a new solution is found, the NonLinearSystem object is updated.
    Avaliable methods are:
    - arclength (ARC)
    """

    def __init__(self, tolerance=1e-5, maxiter: int = 50) -> None:
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.maxRetries = 3
        self.convergence = False
        self.setDefaultAttributes()
        self.output = False
        self.continuedSolution = None
        self.currentSolution = None
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
        eigvals_around_bif = []  # contains eigs just before and after bifurcation
        self.stableBranch = self.checkStability(nls)
        while True:
            solution = nls.get_current_solution()
            parameter = getattr(nls, self.parameterName)
            self.currentSolution = np.append(solution, parameter)
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
                f"Continuation step {i}: {self.parameterName} = {parameter}, avg = {average:.2e}, min = {minimum:.2e}, max = {maximum:.2e}, stable = {self.stableBranch} \n"
                + f"eigvals = {self.eigvals}"
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
                self.print("Bifurcation occured! Determining exact location...")
                if self.previousSolution is not None:
                    bifurcationPoint = self.findBifurcationIndirect(nls)
                else:
                    bifurcationPoint = self.findBifurcationDirect(nls)
                if bifurcationPoint:
                    self.print("Bifurcation point found.")
                    eigvals_around_bif.append((self.previousEigvals, self.eigvals))
                    bifurcationPoints.append(bifurcationPoint)
                else:
                    self.print("Bifurcation point not found.")
                # break  # TODO: implement restart + branch switching

            # save current solution and eigvals previous for next iteration
            self.previousSolution = self.currentSolution
            self.previousEigvals = self.eigvals

        solutions = dict(
            average=np.array(solutionAverages),
            minimum=np.array(solutionMinima),
            maximum=np.array(solutionMaxima),
            parameter=np.array(parameterValues),
            stable=np.array(stableBranch),
            bifurcations=bifurcationPoints,
            eigvals_around_bif=eigvals_around_bif,
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
        self, nls: NonLinearSystem, stepsize, tolerance: float = 1e-4, h: float = 1e-4
    ) -> list:
        solution = self.currentSolution[:-1]
        parameter = self.currentSolution[-1]

        predictorStep = self.predictorStep(nls)

        predictorSolution = solution + 1 * predictorStep[:-1]
        predictorParameter = parameter + 1 * predictorStep[-1]

        # update non-linear system to predictor
        nls.set_current_solution(predictorSolution)
        setattr(nls, self.parameterName, predictorParameter)

        # setup extended system
        extendedSystem = ExtendedSystem(
            nls, self.currentSolution, self.parameterName, stepsize, self.tuneFactor
        )

        # setup rootfinder
        rootfinder = RootFinding(tolerance, self.maxiter)
        rootfinder.output = False

        # corrector iterations
        self.convergence = False
        error = 2 * tolerance
        i = 0
        errors = []
        while error > tolerance and i < self.maxiter:
            # NR rootfinding
            rootfinder.newtonRaphson(extendedSystem, exact=False)

            # update error
            error = np.linalg.norm(nls.evaluate())
            errors.append(error)

            i += 1

        if error <= tolerance:
            self.convergence = True
            currentSolution = nls.get_current_solution()
            currentParameter = getattr(nls, self.parameterName)
            self.continuedSolution = np.append(currentSolution, currentParameter)
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
        # df_dsol = nls.evaluate_derivative()
        # parameter = self.currentSolution[-1]
        # df_dparam = self.derivativeParam(nls, parameter, 1e-4)
        # n = df_dsol.shape[0]
        # k = 0
        # tangentSystem = np.zeros((n + 1, n + 1))
        # while k <= n:  # find index k such that tangentSystem is invertible
        #     e_k = np.zeros(n + 1)
        #     e_k[k] = 1
        #     tangentSystem = np.vstack(
        #         (np.hstack((df_dsol, df_dparam[:, np.newaxis])), [e_k])
        #     )
        #     if np.linalg.matrix_rank(tangentSystem) == n + 1:
        #         break
        #     k += 1

        # return np.linalg.solve(tangentSystem, e_k)
        parameter = self.currentSolution[-1]
        dF_param = self.derivativeParam(nls, parameter, 1e-4)
        dF_sol = nls.evaluate_derivative()

        dsol_dparam = -np.linalg.solve(dF_sol, dF_param)
        dparam_ds = 1 / np.sqrt(1 + self.tuneFactor * np.sum(dsol_dparam**2))
        dsol_ds = dsol_dparam * dparam_ds

        return np.append(dsol_ds, dparam_ds)

    # def predictorStepSecant(self) -> np.ndarray:

    #     return

    # bifurcation detection
    def findBifurcationDirect(self, nls: NonLinearSystem) -> dict:
        """
        Find the exact location of a bifurcation point using a direct method
        see Seydel section 5.4.1 (algorithm 5.4). Only works for bifurcations
        that result in a singular jacobian at the bifurcation point.

        Y = (solution, parameter, h)^T
        F(Y) := branching system
        """
        self.print("Finding bifurcation point using direct method...")
        branching_system = BranchingSystem(nls, self.parameterName)
        # Y = branching_system.get_current_solution()
        # F = branching_system.evaluate()
        rootfinder = RootFinding()
        rootfinder.output = True
        _ = rootfinder.newtonRaphson(branching_system, exact=False)
        if not rootfinder.converged:
            return None
        dF = nls.evaluate_derivative()
        eigvals = np.linalg.eigvals(dF)
        return dict(
            solution=nls.get_current_solution(),
            parameter=getattr(nls, self.parameterName),
            eigvals=eigvals,
            type="...",  # TODO: add type of bifurcation detection
        )

    def findBifurcationIndirect(self, nls: NonLinearSystem) -> dict:
        """
        Find the exact location of a bifurcation point using a secant method
        see Seydel section 5.4.2 (algorithm 5.5).
        """
        self.print("Finding bifurcation point using indirect method...")
        # previousSolution = self.previousSolution[:-1]
        currentSolution = self.currentSolution[:-1]
        previousParameter = self.previousSolution[-1]
        currentParameter = self.currentSolution[-1]
        previousTestFunction = Continuation.testFunction(self.previousEigvals)
        currentTestFunction = Continuation.testFunction(self.eigvals)

        approximateParameter = previousParameter + (
            currentParameter - previousParameter
        ) * previousTestFunction / (previousTestFunction - currentTestFunction)

        # solve for third test function value
        setattr(nls, self.parameterName, approximateParameter)
        rootfinder = RootFinding()
        rootfinder.output = True
        _ = rootfinder.newtonRaphson(nls, exact=True)
        eigvals = np.linalg.eigvals(nls.evaluate_derivative())
        thirdTestFunction = Continuation.testFunction(eigvals)

        # find more exact bifurcation parameter
        zeta = (currentTestFunction - thirdTestFunction) / (
            currentParameter - approximateParameter
        )
        gamma = (
            (previousTestFunction - thirdTestFunction)
            / (previousParameter - approximateParameter)
            - zeta
        ) / (previousParameter - currentParameter)
        correctionPlus = (
            -zeta
            - gamma * (approximateParameter - currentParameter)
            + np.sqrt(
                (zeta + gamma * (approximateParameter - currentParameter)) ** 2
                - 4 * thirdTestFunction * gamma
            )
        ) / (2 * gamma)
        correctionMinus = (
            -zeta
            - gamma * (approximateParameter - currentParameter)
            - np.sqrt(
                (zeta + gamma * (approximateParameter - currentParameter)) ** 2
                - 4 * thirdTestFunction * gamma
            )
        ) / (2 * gamma)
        bifurcationParameter = approximateParameter + correctionPlus
        if (
            bifurcationParameter <= previousParameter
            or bifurcationParameter >= currentParameter
        ):
            bifurcationParameter = approximateParameter + correctionMinus
        setattr(nls, self.parameterName, bifurcationParameter)

        # solve for exact bifurcation point
        rootfinder = RootFinding()
        rootfinder.output = True
        _ = rootfinder.newtonRaphson(nls, exact=True)
        bifurcationPoint = nls.get_current_solution()

        # determine eigvals
        dF = nls.evaluate_derivative()
        eigvals = np.linalg.eigvals(dF)

        # reset to current solution so continuation can continue
        setattr(nls, self.parameterName, currentParameter)
        nls.set_current_solution(currentSolution)

        return dict(
            solution=bifurcationPoint,
            parameter=approximateParameter,
            eigvals=eigvals,
            type="...",  # TODO: add type of bifurcation detection
        )

    def findFirstLimitCycle(self, nls: NonLinearSystem, delta_0: float = 0.02):
        """
        Find a limit cycle starting from a bifurcation point.
        """
        self.nls = nls
        solution = self.nls.get_current_solution()
        parameter = getattr(self.nls, self.parameterName)
        self.currentJacobian = self.nls.evaluate_derivative()
        currentEigs = np.linalg.eigvals(self.currentJacobian)
        self.imaginaryPartHopfEig = np.max(
            [np.imag(eig) for eig in currentEigs if np.imag(eig) != 0]
        )
        # TODO: implement predictor to switch to other branch
        # bsys = BranchingSystem(self.nls, self.parameterName)
        # tangent = bsys.h
        # k = bsys.k
        # y_k = self.nls.get_current_solution()[k]
        # delta = delta_0 * max(1, abs(y_k))
        # predictors = [solution]

        # time span
        t_span = (0, 1)
        t_eval = np.linspace(*t_span, 2)
        self.n = len(solution)

        # initial guess
        y_0 = np.zeros((2 * self.n + 2, t_eval.size))

        # initial guess solution at t=0
        y_0[: self.n, 0] = solution
        y_0[: self.n, 1] = solution

        # initial guess for h1 at t=0
        y_0[self.n] = 1

        # values for parameter
        y_0[2 * self.n] = parameter  # t=0
        # y_0[2 * self.n, 1] = parameter  # t=1

        # period guess
        y_0[2 * self.n + 1, 0] = 2 * np.pi / self.imaginaryPartHopfEig  # at t=0

        # solve boundary value problem
        sol = solve_bvp(self.BVPrhs, self.BVPbcs, t_eval, y_0)
        if sol.status == 0:
            self.print(bcolors.OKGREEN + sol.message + bcolors.ENDC)
        else:
            self.print(bcolors.WARNING + sol.message + bcolors.ENDC)

        return sol

    def findFirstLimitCycle2(self, nls: NonLinearSystem, delta_0: float = 0.02):
        """
        Find a limit cycle starting from a bifurcation point.
        """
        self.nls = nls
        solution = self.nls.get_current_solution()
        parameter = getattr(self.nls, self.parameterName)
        self.currentJacobian = self.nls.evaluate_derivative()
        currentEigs = np.linalg.eigvals(self.currentJacobian)
        self.imaginaryPartHopfEig = np.max(
            [np.imag(eig) for eig in currentEigs if np.imag(eig) != 0]
        )
        self.currentF = self.nls.evaluate()
        # TODO: implement predictor to switch to other branch
        # bsys = BranchingSystem(self.nls, self.parameterName)
        # tangent = bsys.h
        # k = bsys.k
        # y_k = self.nls.get_current_solution()[k]
        # delta = delta_0 * max(1, abs(y_k))
        # predictors = [solution]

        # time span
        t_span = (0, 1)
        t_eval = np.linspace(*t_span, 2)
        self.n = len(solution)

        # initial guess
        y_0 = np.zeros((2 * self.n + 2, t_eval.size))

        # initial guess solution at t=0
        y_0[: self.n, 0] = solution
        # y_0[: self.n,1] = solution

        # initial guess for h1 at t=0
        y_0[self.n : self.n * 2, 0] = 1

        # values for parameter
        y_0[2 * self.n, 0] = parameter  # t=0
        # y_0[2 * self.n, 1] = parameter  # t=1

        # period guess
        y_0[2 * self.n + 1, 0] = 18  # 2 * np.pi / self.imaginaryPartHopfEig  # at t=0

        # solve boundary value problem
        sol = solve_bvp(self.BVPrhs, self.BVPbcs, t_eval, y_0)
        if sol.status == 0:
            self.print(bcolors.OKGREEN + sol.message + bcolors.ENDC)
        else:
            self.print(bcolors.WARNING + sol.message + bcolors.ENDC)

        return sol

    def BVPrhs(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        solution = y[: self.n]
        h = y[self.n : 2 * self.n]
        parameter = y[2 * self.n]
        period = y[2 * self.n + 1]

        self.nls.set_current_solution(solution)
        setattr(self.nls, self.parameterName, parameter)
        f = self.nls.evaluate()

        df = self.nls.evaluate_derivative_vectorized()

        fh = np.einsum("ikj, ik ->ij", df, h.T)

        k = t.size
        zero = np.zeros(k)
        return np.vstack((period * f, period * fh.T, zero, zero))

    def BVPbcs(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Residual of the periodic boundary conditions
        """
        solution_a = ya[: self.n]
        h_a = ya[self.n : 2 * self.n]
        solution_b = yb[: self.n]
        h_b = yb[self.n : 2 * self.n]
        parameter_a = ya[2 * self.n]
        # setattr(self.nls, self.parameterName, parameter_a)
        # self.nls.set_current_solution(solution_a)
        # df = self.nls.evaluate_derivative()
        df = self.currentJacobian
        return np.concatenate(
            (solution_a - solution_b, h_a - h_b, [np.sum(df[0] * h_a)], [h_a[0] - 1])
        )

    def findNextLimitCycle(self, nls: NonLinearSystem, period_guess: float):
        """
        Find a nearby limit cycle starting from a bifurcation or any cycle on the corresponding branch.
        """
        self.nls = nls
        self.oldSolution = self.nls.get_current_solution()
        self.oldParameter = getattr(self.nls, self.parameterName)
        currentJacobian = self.nls.evaluate_derivative()
        self.currentF = self.nls.evaluate()
        currentEigs = np.linalg.eigvals(currentJacobian)
        self.imaginaryPartHopfEig = np.max(
            [np.imag(eig) for eig in currentEigs if np.imag(eig) != 0]
        )

        # time span
        t_span = (0, 1)
        t_eval = np.linspace(*t_span, 2)
        self.n = len(self.oldSolution)

        # initial guess
        y_0 = np.zeros((self.n + 2, t_eval.size))

        # initial guess solution at t=0
        y_0[: self.n, 0] = self.oldSolution
        y_0[: self.n, 1] = self.oldSolution

        # initial guess for the period
        y_0[self.n, 0] = period_guess

        # initial guess for the parameter
        y_0[self.n + 1] = self.oldParameter

        # solve boundary value problem
        sol = solve_bvp(self.BVPrhs2, self.BVPbcs2, t_eval, y_0)
        if sol.status == 0:
            self.print(bcolors.OKGREEN + sol.message + bcolors.ENDC)
        else:
            self.print(bcolors.WARNING + sol.message + bcolors.ENDC)

        return sol

    def BVPrhs2(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        solution = y[: self.n]
        # period = y[self.n]
        parameter = y[self.n]
        setattr(self.nls, self.parameterName, parameter)
        self.nls.set_current_solution(solution)
        f = self.nls.evaluate()
        k = t.size
        zero = np.zeros(k)
        # return np.vstack((f, zero, zero))
        return np.vstack((f, zero, zero))

    def BVPbcs2(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Residual of the periodic boundary conditions
        """
        solution_a = ya[: self.n]
        solution_b = yb[: self.n]
        parameter_a = ya[self.n]
        parameter_b = yb[self.n]
        setattr(self.nls, self.parameterName, parameter_a)
        self.nls.set_current_solution(solution_a)
        f = self.nls.evaluate()
        # return np.concatenate(
        #     (solution_a - solution_b, [self.BVPphase(solution_a)], [f[0]])
        # )
        return np.concatenate(
            (solution_a - solution_b, [f[0], self.BVPphase(solution_a)])
        )

    def BVPphase(self, y: np.ndarray) -> float:
        """
        Residual of the phase condition
        """
        return (y - self.oldSolution) @ self.currentF

    # helper methods
    @staticmethod
    def testFunction(eigvals: np.ndarray) -> float:
        """
        Calculate the test function for bifurcation detection.
        """
        return np.max(np.real(eigvals))

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

    def shootingMethod(
        self,
        nls: NonLinearSystem,
        type: str,  # specify if continuation or switching
        period_guess: float = None,
        tolerance=1e-12,
        maxiter=200,
        stepsize=None,
    ):
        """
        shooting method for finding limit cycles
        """
        # get current solution and parameter
        self.nls = nls
        solution = self.nls.get_current_solution()
        parameter = getattr(self.nls, self.parameterName)
        self.oldParameter = parameter
        self.n = len(solution)

        # objective function for the shooting method
        if type == "cont":
            objective = self.shootingMethodObjectiveCont
            y_0 = np.zeros(self.n + 2)
            y_0[: self.n] = solution
            y_0[self.n] = parameter
            y_0[self.n + 1] = period_guess
            if stepsize is None:
                raise ValueError("Stepsize required for periodic branch cont.")
            self.pcont_stepsize = stepsize
        elif type == "hopf-switch":
            objective = self.shootingMethodObjectiveHopfSwitch
            y_0 = np.zeros(2 * self.n + 2)
            y_0[: self.n] = solution
            y_0[self.n : self.n * 2] = 0
            y_0[self.n] = 1
            y_0[2 * self.n] = parameter  # t=0
            y_0[2 * self.n + 1] = period_guess
        elif type == "pdouble-switch":
            objective = self.shootingMethodObjectivePdoubleSwitch
            y_0 = np.zeros(2 * self.n + 2)
            y_0[: self.n] = solution
            y_0[self.n : self.n * 2] = 0
            y_0[self.n] = 1
            y_0[2 * self.n] = parameter  # t=0
            y_0[2 * self.n + 1] = period_guess
        else:
            raise ValueError("Type not recognized. Choose 'cont' or 'switch'.")

        # check if period guess is provided
        if period_guess is None:
            raise ValueError("Period guess required for switching continuation.")

        # define time span
        self.t_span = (0, 1)

        out, infodict, ier, msg = fsolve(
            objective,
            y_0,
            xtol=tolerance,
            maxfev=maxiter,
            full_output=True,
        )
        if ier == 1:
            # pprint(infodict)
            self.print(bcolors.OKGREEN + msg + bcolors.ENDC)
        else:
            self.print(bcolors.WARNING + msg + bcolors.ENDC)
        return out, ier

    def shootingMethodRHSHopfSwitch(self, t: float, y: np.ndarray):
        """
        RHS of the shooting method
        """
        solution = y[: self.n]
        h = y[self.n : 2 * self.n]
        parameter = y[2 * self.n]
        period = y[2 * self.n + 1]
        setattr(self.nls, self.parameterName, parameter)
        self.nls.set_current_solution(solution)
        f = self.nls.evaluate()
        df = self.nls.evaluate_derivative()
        return np.concatenate(
            # (period * f, period * np.einsum("ij, j -> i", df, h), [0, 0])
            (period * f, period * df @ h, [0, 0])
        )

    def shootingMethodObjectiveHopfSwitch(self, y0):
        """
        Objective function for the shooting method
        """
        # save initial state
        solution0 = y0[: self.n]
        h0 = y0[self.n : 2 * self.n]
        parameter0 = y0[2 * self.n]
        period0 = y0[2 * self.n + 1]
        self.nls.set_current_solution(solution0)
        setattr(self.nls, self.parameterName, parameter0)
        df = self.nls.evaluate_derivative()

        # integrate
        sol = solve_ivp(
            self.shootingMethodRHSHopfSwitch,
            self.t_span,
            y0,
            vectorized=False,
            max_step=0.01,
        )

        # check residuals
        y1 = sol.y
        solution1 = y1[: self.n, -1]
        h1 = y1[self.n : 2 * self.n, -1]
        parameter1 = y1[2 * self.n, -1]
        period1 = y1[2 * self.n + 1, -1]

        return np.concatenate(
            (solution0 - solution1, h0 - h1, [np.sum(h0 * df[0]), h0[0] - 1])
            # (h0 - h1, solution0 - solution1, [np.sum(h0 * df[0]), parameter0 - self.oldParameter - 0.02])
        )

    def shootingMethodRHSCont(self, t: float, y: np.ndarray):
        """
        RHS of the shooting method
        """
        solution = y[: self.n]
        parameter = y[self.n]
        period = y[self.n + 1]
        setattr(self.nls, self.parameterName, parameter)
        self.nls.set_current_solution(solution)
        f = self.nls.evaluate()
        # df = self.nls.evaluate_derivative()
        return np.concatenate((period * f, [0, 0]))

    def shootingMethodObjectiveCont(self, y0):
        """
        Objective function for the shooting method
        """
        # save initial state
        solution0 = y0[: self.n]
        parameter0 = y0[self.n]
        # period0 = y0[self.n + 1]
        self.nls.set_current_solution(solution0)
        setattr(self.nls, self.parameterName, parameter0)
        f = self.nls.evaluate()
        # df = self.nls.evaluate_derivative()

        # integrate
        sol = solve_ivp(
            self.shootingMethodRHSCont,
            self.t_span,
            y0,
            vectorized=False,
            max_step=0.005,
        )

        # check residuals
        y1 = sol.y
        solution1 = y1[: self.n, -1]
        parameter1 = y1[self.n, -1]
        # period1 = y1[self.n + 1, -1]

        return np.concatenate(
            # (solution0 - solution1, [f[0], parameter0-self.oldParameter-stepsize])
            (
                solution0 - solution1,
                [parameter0 - self.oldParameter - self.pcont_stepsize, f[0]],
            )
        )

    def shootingMethodRHSPdoubleSwitch(self, t: float, y: np.ndarray):
        """
        RHS of the shooting method
        """
        solution = y[: self.n]
        h = y[self.n : 2 * self.n]
        parameter = y[2 * self.n]
        period = y[2 * self.n + 1]
        setattr(self.nls, self.parameterName, parameter)
        self.nls.set_current_solution(solution)
        f = self.nls.evaluate()
        df = self.nls.evaluate_derivative()
        return np.concatenate(
            # (period * f, period * np.einsum("ij, j -> i", df, h), [0, 0])
            (period * f, period * df @ h, [0, 0])
        )

    def shootingMethodObjectivePdoubleSwitch(self, y0):
        """
        Objective function for the shooting method
        """
        # save initial state
        solution0 = y0[: self.n]
        h0 = y0[self.n : 2 * self.n]
        parameter0 = y0[2 * self.n]
        # period0 = y0[2 * self.n + 1]
        self.nls.set_current_solution(solution0)
        setattr(self.nls, self.parameterName, parameter0)
        f = self.nls.evaluate()
        # df = self.nls.evaluate_derivative()

        # integrate
        sol = solve_ivp(
            self.shootingMethodRHSPdoubleSwitch,
            self.t_span,
            y0,
            vectorized=False,
            max_step=0.001,
        )

        # check residuals
        y1 = sol.y
        solution1 = y1[: self.n, -1]
        h1 = y1[self.n : 2 * self.n, -1]
        # parameter1 = y1[2 * self.n, -1]
        # period1 = y1[2 * self.n + 1, -1]

        return np.concatenate(
            (solution0 - solution1, h0 + h1, [f[0], h0[0] - 1])
            # (h0 - h1, solution0 - solution1, [np.sum(h0 * df[0]), parameter0 - self.oldParameter - 0.02])
        )

    def monodromyMatrix(
        self,
        cycle_point: np.ndarray,
        cycle_parameter: np.ndarray,
        cycle_period: float,
    ) -> np.ndarray:
        """
        Calculate the monodromy matrix of a limit cycle.
        """
        n = len(cycle_point)
        self.n = n
        t_span = (0, 1)
        t_eval = np.linspace(*t_span, 300)

        # obtain columns of monodromy matrix
        monodromy = np.zeros((n, n))
        for j in range(n):
            y_0 = np.zeros(2 * n + 2)
            y_0[:n] = cycle_point
            y_0[n + j] = 1
            y_0[2 * n] = cycle_parameter
            y_0[2 * n + 1] = cycle_period
            sol = solve_ivp(
                self.shootingMethodRHSHopfSwitch,
                t_span,
                y_0,
                t_eval=t_eval,
                vectorized=False,
                max_step=0.001,
            )
            h = sol.y[n : 2 * n, -1]
            monodromy[:, j] = h
        return monodromy

    def calculateLimitCycleStability(
        self,
        nls: NonLinearSystem,
        point: np.ndarray,
        parameter: float,
        period,
        tolerance=1e-3,
    ) -> tuple[np.ndarray, bool]:
        self.nls = nls
        monodromy = self.monodromyMatrix(point, parameter, period)
        eig = np.linalg.eigvals(monodromy)
        # check for approximate unity eigenvalues
        unityEigIndex = np.isclose(np.abs(eig), 1, atol=tolerance)
        if np.all(np.abs(eig[~unityEigIndex]) < 1):
            return eig, True
        else:
            return eig, False

    def calculateFloquetMultipliers(self, monodromy_matrix: np.ndarray):
        """
        Calculate the Floquet multipliers of a limit cycle.
        """
        pass

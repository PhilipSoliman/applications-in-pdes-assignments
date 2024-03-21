import numpy as np
from tqdm import tqdm
from non_linear_system import NonLinearSystem
from root_finding import RootFinding


class Continuation(NonLinearSystem):

    def __init__(self, maxiter: int) -> None:
        self.maxiter = maxiter
        self.convergence = False
        self.output = False
        self.method = None
        self.pbar = None
        self.current_solution = None
        self.continued_solution = None

    def execute(self, nls: NonLinearSystem, stepsize: float) -> None:
        self.converged = False
        if self.current_solution is None:
            raise ValueError(
                "Current solution not set. First specify initial root of the (unparameterized) system."
            )

        if self.method == "ARC":
            continued_solution = self.arclengthUpdate(nls, stepsize)
        else:
            raise ValueError("Method not specified and/or implemented.")

        if self.convergence:
            return continued_solution
        else:
            print(
                f"'{self.method}' continuation did not converge within the maximum number of iterations. Exiting..."
            )
            return self.current_solution

    def arclengthPC(
        self, nls: NonLinearSystem, stepsize, tolerance: float = 1e-5, h: float = 1e-4
    ) -> np.ndarray:

        # predictor step
        dF_param = self.derivativeParam(nls, h)
        dF_sol = nls.evaluate_derivative()

        dsol_dparam = np.linalg.solve(dF_sol, dF_param)
        dparam_ds = 1 / np.sqrt(1 + self.tune_factor * np.linalg.norm(dsol_dparam))
        dsol_ds = dsol_dparam * dparam_ds

        newSolution = self.current_solution + stepsize * dsol_ds
        newParam = getattr(nls, self.parameter) + stepsize * dparam_ds

        # corrector iterations
        self.convergence = False
        error = 2 * tolerance
        while error > tolerance:
            dF_param = self.derivativeParam(nls, h)
            dF_sol = nls.evaluate_derivative()

            z1 = np.linalg.solve(dF_sol, -nls.evaluate())
            z2 = np.linalg.solve(dF_sol, dF_param)

            dp_dsol = (
                2 * self.tune_factor * (newSolution - self.current_solution)[:, None]
            )
            dp_dparam = (
                2 * (1 - self.tune_factor) * (newParam - getattr(nls, self.parameter))
            )

            p = (
                self.tuning_factor
                * np.linalg.norm(newSolution - self.current_solution) ** 2
                + (1 - self.tuning_factor)
                * (newParam - getattr(nls, self.parameter)) ** 2
                - (stepsize**2)
            )

            dparam = (-p - dp_dsol.T @ z1) / (dp_dparam - dp_dsol.T @ z2)
            dsol = z1 + z2 * dparam

            newSolution = newSolution + dsol
            newParam = newParam + dparam

            error = np.linalg.norm(self.current_solution)

        if error <= tolerance:
            self.convergence = True
            self.current_solution = newSolution
            setattr(nls, self.parameter, newParam)
            
        return self.current_solution

    def arclength(
        self,
        nls: NonLinearSystem,
        parameter: str,
        stepsize: float,
        tune_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Perform continuation using the arclength method. Calls main method
        using the specified NonLinearSystem and parameter.
        """
        if not hasattr(nls, parameter):
            raise ValueError(
                f"Parameter {parameter} not found in given NonLinearSystem."
            )
        self.parameter = parameter
        self.method = "ARC"
        self.tune_factor = tune_factor
        return self.execute(nls, parameter, stepsize, tune_factor)

    def derivativeParam(self, nls: NonLinearSystem, h: float = 1e-6) -> np.ndarray:
        """
        Calculate the derivative of the non-linear system w.r.t. the parameter.
        """
        self.F = nls.evaluate()
        setattr(nls, self.parameter, getattr(nls, self.parameter) + h)
        dF_param = (nls.evaluate() - F) / h

        # reset parameter
        setattr(nls, self.parameter, getattr(nls, self.parameter) - h)
        return dF_param

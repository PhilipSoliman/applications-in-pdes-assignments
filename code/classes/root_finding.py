import numpy as np
from tqdm import tqdm
from non_linear_system import NonLinearSystem


class RootFinding:

    def __init__(self, tolerance: float = 1e-5, maxiter: int = 100) -> None:
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.converged = False
        self.output = False
        self.method = None
        self.pbar = None
        self.singular_matrix = False

    # General method for finding roots
    def findRoot(
        self, nls: NonLinearSystem, exact: bool = True, stepsize: float = 1e-6
    ) -> list:
        F = nls.evaluate()
        error = np.linalg.norm(F)
        errors = []
        i = 0
        self.converged = False
        self.print(f"Running {self.method}...")
        self.instantiate_pbar(self.maxiter)
        while error > self.tolerance:
            self.update_pbar(i, error)
            if i == self.maxiter:
                self.close_pbar()
                self.print(
                    "{self.method} method did not converge within the maximum number of iterations. Exiting..."
                )
                break

            if self.method == "NR":
                F = self.newtonRaphsonUpdate(nls, F, exact, stepsize=stepsize)
            elif self.method == "BR":
                if i == 0:
                    B = nls.evaluate_derivative()
                F, B = self.broydensMethodUpdate(nls, F, B)
            else:
                raise ValueError("Method not specified and/or implemented.")

            # check for singular matrix
            if self.singular_matrix:
                self.close_pbar()
                self.print(f"Singular matrix encountered at iteration {i}. Exiting...")
                break

            # calculate current error
            error = np.linalg.norm(F)
            errors.append(error)

            # set iteration counter
            i += 1

        if error <= self.tolerance:
            self.close_pbar()
            self.converged = True
            self.print(f"{self.method} converged.")

        return errors

    # Method updates
    def newtonRaphsonUpdate(
        self, nls: NonLinearSystem, F: np.ndarray, exact: bool, stepsize: float = 1e-6
    ) -> np.ndarray:
        if exact:
            dF = nls.evaluate_derivative()
        else:
            dF = nls.evaluate_derivative_finite_difference(stepsize)
        update = np.linalg.solve(dF, F)
        nls.update_solution(-update)
        F = nls.evaluate()
        return F

    def broydensMethodUpdate(
        self, nls: NonLinearSystem, F: np.ndarray, B: np.ndarray
    ) -> tuple[np.ndarray]:
        try:
            update = -np.linalg.solve(B, F)
        except np.linalg.LinAlgError:
            self.singular_matrix = True
            return F, B
        nls.update_solution(update)  # TODO: check if solution is implicitly updated
        F = nls.evaluate()
        B += np.outer(F, update) / (update.T @ update)
        return F, B

    # Method handles
    def newtonRaphson(
        self, NLS: NonLinearSystem, exact: bool = True, stepsize: float = 1e-6
    ) -> list:
        self.method = "NR"
        return self.findRoot(NLS, exact, stepsize)

    def broydensMethod(
        self, NLS: NonLinearSystem, exact: bool = True, stepsize: float = 1e-6
    ) -> list:
        self.method = "BR"
        return self.findRoot(NLS, exact, stepsize)

    # Control output
    def print(self, message: str, **kwargs) -> None:
        if self.output:
            print(message, **kwargs)

    def instantiate_pbar(self, maxiter: int) -> None:
        if self.output:
            self.pbar = tqdm(
                desc="", total=maxiter, unit=f"{self.method} update", miniters=5
            )

    def update_pbar(self, i: int, error: float) -> None:
        if self.output:
            self.pbar.set_description(
                f"iteration: {i}, error: {error:.2e}, tolerance: {self.tolerance:.2e}, maxiter: {self.maxiter}"
            )

    def close_pbar(self) -> None:
        if self.output:
            self.pbar.close()

    def arcLength() -> None:
        pass

    # def newtonRaphson(self, NLS: NonLinearSystem, exact: bool, stepsize: float) -> list:
    #     error = np.linalg.norm(NLS.evaluate())
    #     errors = []
    #     i = 0
    #     if exact:
    #         self.print(f"Running Newton-Raphson with exact derivatives...")
    #     else:
    #         self.print(
    #             f"Running Newton-Raphson with finite difference derivatives (h = {stepsize})..."
    #         )
    #     pbar = tqdm(desc="", total=self.maxiter, unit="NR update", miniters=5)
    #     while error > self.tolerance:
    #         pbar.set_description(
    #             f"iteration: {i}, error: {error:.2e}, tolerance: {self.tolerance:.2e}, maxiter: {self.maxiter}"
    #         )
    #         if i == self.maxiter:
    #             pbar.close()
    #             self.print(
    #                 "Newton-Raphson method did not converge within the maximum number of iterations. Exiting..."
    #             )
    #             break

    #         F = NLS.evaluate()
    #         if exact:
    #             dF = NLS.evaluate_derivative()
    #         else:
    #             dF = NLS.evaluate_derivative_finite_difference(stepsize)
    #         update = np.linalg.solve(dF, F)
    #         NLS.update_solution(-update)

    #         # calculate current error
    #         error = np.linalg.norm(F)
    #         errors.append(error)

    #         # set iteration counter
    #         i += 1

    #     if error <= self.tolerance:
    #         pbar.close()
    #         self.converged = True
    #         self.print("Newton-Raphson converged.")

    #     return errors

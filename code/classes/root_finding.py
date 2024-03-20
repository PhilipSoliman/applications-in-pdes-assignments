import numpy as np
from tqdm import tqdm
from ebm import NonLinearSystem


class RootFinding:

    def __init__(self, tolerance: float = 1e-5, maxiter: int = 100) -> None:
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.converged = False
        self.output = False

    def newtonRaphson(
        self, NLS: NonLinearSystem, exact: bool = True, stepsize: float = 1e-6
    ) -> list:
        error = np.linalg.norm(NLS.evaluate())
        errors = []
        i = 0
        if exact:
            self.print(f"Running Newton-Raphson with exact derivatives...")
        else:
            self.print(
                f"Running Newton-Raphson with finite difference derivatives (h = {stepsize})..."
            )
        pbar = tqdm(desc="", total=self.maxiter, unit="NR update", miniters=5)
        while error > self.tolerance:
            pbar.set_description(
                f"iteration: {i}, error: {error:.2e}, tolerance: {self.tolerance:.2e}, maxiter: {self.maxiter}"
            )
            if i == self.maxiter:
                pbar.close()
                self.print(
                    "Newton-Raphson method did not converge within the maximum number of iterations. Exiting..."
                )
                break

            F = NLS.evaluate()
            if exact:
                dF = NLS.evaluate_derivative()
            else:
                dF = NLS.evaluate_derivative_finite_difference(stepsize)
            # print(dF)
            update = np.linalg.solve(dF, F)
            NLS.update_solution(-update)

            # calculate current error
            error = np.linalg.norm(F)
            errors.append(error)

            # set iteration counter
            i += 1

        if error <= self.tolerance:
            pbar.close()
            self.converged = True
            self.print("Newton-Raphson converged.")

        return errors

    def broydensMethod() -> None:
        pass

    def arcLength() -> None:
        pass

    def print(self, message: str) -> None:
        if self.output:
            print(message)

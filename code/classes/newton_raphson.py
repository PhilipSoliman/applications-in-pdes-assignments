import numpy as np
from typing import Callable
from tqdm import tqdm


class NewtonRaphson:

    def __init__(self, tolerance: float = 1e-6, maxiter: int = 100) -> None:
        self.tolerance = tolerance
        self.maxiter = maxiter

    def run(self, NLS) -> list:
        e_k = 2 * self.tolerance
        errors = [e_k]
        i = 0
        print("Running Newton-Raphson method...")
        pbar = tqdm(desc="", total=self.maxiter, unit="NR update", miniters=5)
        while e_k > self.tolerance:
            pbar.set_description(
                f"iteration: {i}, error: {errors[i]:.2e}, tolerance: {self.tolerance:.2e}, maxiter: {self.maxiter}"
            )
            if i == self.maxiter:
                pbar.close()
                print(
                    "Newton-Raphson method did not converge within the maximum number of iterations. Exiting..."
                )
                break

            # perform coefficient update
            F_k = NLS.evaluate()
            F_t_k = NLS.evaluate_derivative()
            update = np.linalg.solve(F_t_k, F_k)
            NLS.T_coeffs -= update

            # calculate current error
            e_k = np.linalg.norm(F_k)
            errors.append(e_k)

            # set iteration counter
            i += 1

        if e_k <= self.tolerance:
            pbar.close()
            print("Newton-Raphson converged.")

        return errors

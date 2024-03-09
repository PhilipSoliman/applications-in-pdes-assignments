import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from custom_pyutils import pyutils
from typing import Callable
from timeit import timeit
from non_linear_system import NonLinearSystem
from tqdm import tqdm


# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n = 10  # number of Legendre polynomials
number_of_quad_points = 2 * n  # number of quadrature points
grid_resolution = 100  # resolution of the grid
tolerance = 1e-6  # tolerance for Newton-Raphson method

############### Setup Non-linear System of Equations ####################
NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

print("Problem constants:\n  ", end="")
pprint({k: v for k, v in NLS.__dict__.items()})

print("Plotting initial guess for T...")
T_initial = NLS.T_x(NLS.x)
plt.close()
plt.plot(NLS.x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)
print("Saved initial guess for T to:\n  ", fn)

################### Newton-Raphson Method ####################
errors = [2 * tolerance]
maxiter = 100
i = 0  # iteration counter
print("Running Newton-Raphson method...")
pbar = tqdm(desc="", total=maxiter, unit="NR update", miniters=1)
while errors[i] > tolerance:
    if i == maxiter:
        print(
            "Newton-Raphson method did not converge within the maximum number of iterations. Exiting..."
        )
        break

    errors.append(
        np.linalg.norm(NLS.evaluate() - NLS.T_coeffs)
    )  # error is the norm of the residual
    # TODO: fix the coefficient update (probably something wrong with the corr. getter/setter)
    NLS.T_coeffs = NLS.T_coeffs - np.linalg.solve(
        NLS.evaluate_derivative(), NLS.T_coeffs
    )  # update T_coeff
    i += 1  # increment iteration counter
    pbar.set_description(
        f"iteration: {i}, error: {errors[i]:.2e}"
    )  # update progress bar

print(NLS.T_coeffs)

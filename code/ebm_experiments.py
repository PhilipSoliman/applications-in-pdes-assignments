import numpy as np
import matplotlib.pyplot as plt
from helper import pyutils

pyutils.add_modules_to_path()
from ebm import EBM
from root_finding import RootFinding


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
initial_temperature = (
    220  # 220  # initial guess for (T # follows from boundary conditions and delta = 0)
)
maxiter = 100  # maximum number of iterations for Newton-Raphson method

############### Setup Non-linear System of Equations ####################
ebm = EBM(n, number_of_quad_points, grid_resolution)

# print("Problem constants:\n  ", end="")
# pprint({k: v for k, v in ebm.__dict__.items()})

print("Plotting initial guess for T...")
ebm.T_coeffs[0] = initial_temperature
T_initial = ebm.T_x(ebm.x)
plt.close()
plt.plot(ebm.x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)
print("Saved initial guess for T to:\n  ", fn)

################### Newton-Raphson Method ####################
rootfinding = RootFinding(maxiter=maxiter)

# run Newton-Raphson method
errors = rootfinding.newtonRaphson(ebm)
T_final_exact = ebm.T_x(ebm.x)

# run Newton-Raphson method with finite difference
ebm.T_coeffs = np.zeros(n)
ebm.T_coeffs[0] = initial_temperature
errors_fd = rootfinding.newtonRaphson(ebm, exact=False, stepsize=1e3)
T_final_fd = ebm.T_x(ebm.x)

# plot error convergence
plt.close()
plt.plot(errors)
plt.plot(errors_fd)
plt.title("Error convergence")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale("log")
fn = output_dir / "error_convergence.png"
plt.savefig(fn, dpi=500)
print("Saved error convergence to:\n  ", fn)

# plot final solution
plt.close()
plt.plot(ebm.x, T_final_exact, label="Exact Jacobian")
plt.plot(ebm.x, T_final_fd, label="Finite Difference Jacobian")
plt.title("Final solution for T")
plt.legend()
fn = output_dir / "equilibrium_T.png"
plt.savefig(fn, dpi=500)
print("Saved equilibrium T to:\n  ", fn)

##################### exact vs finite difference jacobian #####################
stepsizes = np.logspace(-13, 7, 100)
errors = []
ebm.T_coeffs = np.zeros(n)
ebm.T_coeffs[0] = initial_temperature
_ = rootfinding.newtonRaphson(ebm) # find equilibrium solution
jacobian_exact = ebm.evaluate_derivative()
print("Running finite difference jacobian approximation analysis...")
for h in stepsizes:
    jacobian_fd = ebm.evaluate_derivative_finite_difference(h)
    error = np.linalg.norm(jacobian_exact-jacobian_fd, ord='fro')
    errors.append(error)

plt.close()
plt.plot(stepsizes, errors)
plt.xscale("log")
plt.yscale("log")
plt.title("Error in Jacobian approximation")
plt.xlabel("Stepsize")
plt.ylabel("Error")
fn = output_dir / "jacobian_approximation_error.png"
plt.savefig(fn, dpi=500)
print("Saved jacobian approximation error to:\n  ", fn)
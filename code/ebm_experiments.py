import numpy as np
import matplotlib.pyplot as plt
from helper import pyutils
pyutils.add_modules_to_path()
from non_linear_system import NonLinearSystem
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
initial_temperature = 280 #220  # initial guess for (T # follows from boundary conditions and delta = 0)
maxiter = 1000  # maximum number of iterations for Newton-Raphson method

############### Setup Non-linear System of Equations ####################
NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

# print("Problem constants:\n  ", end="")
# pprint({k: v for k, v in NLS.__dict__.items()})

print("Plotting initial guess for T...")
NLS.T_coeffs[0] = initial_temperature 
T_initial = NLS.T_x(NLS.x)
plt.close()
plt.plot(NLS.x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)
print("Saved initial guess for T to:\n  ", fn)

################### Newton-Raphson Method ####################
print('legendre_norms * legendre_eigs / D', NLS.legendre_norms * NLS.legendre_eigs / NLS.D)
rootfinding = RootFinding(maxiter=maxiter)
errors = rootfinding.newtonRaphson(NLS)

# plot error convergence
plt.close()
plt.plot(errors[1:])
plt.title("Error convergence")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.yscale("log")
fn = output_dir / "error_convergence.png"
plt.savefig(fn, dpi=500)
print("Saved error convergence to:\n  ", fn)


# plot final solution
T_final = NLS.T_x(NLS.x)
plt.close()
plt.plot(NLS.x, T_final, label="Final solution")
plt.title("Final solution for T")
fn = output_dir / "equilibrium_T.png"
plt.savefig(fn, dpi=500)
print(NLS.T_coeffs)
print("Saved equilibrium T to:\n  ", fn)

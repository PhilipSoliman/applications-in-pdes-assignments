import numpy as np
import matplotlib.pyplot as plt
from helper import pyutils

pyutils.add_modules_to_path()
from ebm import EBM
from root_finding import RootFinding

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n_polys = 5  # number of Legendre polynomials
n_quads = 2 * n_polys  # number of quadrature points
grid_resolution = 100  # resolution of the grid
initial_temperature = 225

############### Setup Non-linear System of Equations #####
ebm = EBM(n_polys, n_quads, grid_resolution)

## root finding
rootfinding = RootFinding(maxiter=1000)
rootfinding.output = True
ebm.T_coeffs[0] = initial_temperature
errors_nr = rootfinding.newtonRaphson(ebm, initial_temperature)
ebm.T_coeffs[0] = initial_temperature
errors_br = rootfinding.broydensMethod(ebm, initial_temperature)

# plot errors
plt.plot(errors_nr, label="Newton-Raphson")
plt.plot(errors_br, label="Broyden's method")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend()
plt.title("Convergence of Newton-Raphson and Broyden's method")
fn = output_dir / f"convergence_NR_BR.png"
plt.savefig(fn, dpi=500)
print("Saved convergence plot to:\n  ", fn)

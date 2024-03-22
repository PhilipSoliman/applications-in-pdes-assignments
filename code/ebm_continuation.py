import numpy as np
import matplotlib.pyplot as plt
from helper import pyutils

pyutils.add_modules_to_path()
from ebm import EBM
from root_finding import RootFinding
from continuation import Continuation

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n_polys = 5  # number of Legendre polynomials
n_quads = 2 * n_polys  # number of quadrature points
grid_resolution = 100  # resolution of the grid
initial_temperature = 220.0  # initial temperature

############### Setup Non-linear System of Equations #####
ebm = EBM(n_polys, n_quads, grid_resolution)
ebm.T_coeffs[0] = initial_temperature

############### Continuation parameters ##################
parameter_name = "mu"  # parameter to be continued
method = "arclength"  # continuation method
stepsize = 0.001  # stepsize (needs to be small, why?)
tune_factor = 0.01  # tune factor (needs to be small, why?)
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations

############### Continuation ############################
# find initial solution
rootfinding = RootFinding(tolerance, maxiter)
rootfinding.output = True
rootfinding.newtonRaphson(ebm)
print(f"Initial solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")
# plt.plot(ebm.x, ebm.T_x(ebm.x), label="Initial guess")

continuation = Continuation(tolerance, maxiter)
errors = continuation.arclength(ebm, parameter_name, stepsize, tune_factor)
plt.plot(errors)
plt.yscale("log")
plt.show()
print(f"Continuation solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")

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
initial_temperature = 225.0  # initial temperature

############### Setup Non-linear System of Equations #####
ebm = EBM(n_polys, n_quads, grid_resolution)

############### Continuation parameters ##################
maxiter = 100  # maximum number of iterations
parameter_name = "mu"  # parameter to be continued
method = "arclength"  # continuation method
stepsize = 0.1  # stepsize
tune_factor = 0.5  # tune factor
tolerance = 1e-5  # tolerance

############### Continuation ############################
# find initial solution
rootfinding = RootFinding(maxiter)
rootfinding.output = True
rootfinding.newtonRaphson(ebm, initial_temperature, tolerance)
print(f"Initial solution {ebm.T_avg()}")

continuation = Continuation(maxiter)
errors = continuation.arclength(ebm, parameter_name, stepsize, tune_factor, tolerance)
plt.plot(errors)
plt.show()
print(f"Continuation solution {ebm.T_avg()}")
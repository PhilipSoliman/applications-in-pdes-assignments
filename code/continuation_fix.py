import matplotlib.pyplot as plt
import numpy as np
from helper import pyutils

pyutils.add_modules_to_path()
from continuation import Continuation
from ebm import EBM
from root_finding import RootFinding

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# set standard matplotlib style
pyutils.set_style()

############### Setup Non-linear System of Equations #####
n_polys = 5  # number of Legendre polynomials
n_quads = 2 * n_polys  # number of quadrature points
grid_resolution = 100  # resolution of the grid
ebm = EBM(n_polys, n_quads, grid_resolution)

############### Continuation parameters ##################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
parameter_name = "mu"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0, 100)  # range of the parametervalues of interest
stepsize = 0.1  # 0.1  # stepsize (needs to be small, why?)
tune_factor = 0.001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 100  # maximum number of continuations

############### Rootfinder settings #################
rootfinding = RootFinding(tolerance, maxiter)
rootfinding.output = True

############### Continuation (loop) ##########################
print("Building bifurcation diagram (continuation loop)...")
ebm.D = 0.3
ebm.T_coeffs[0] = 225
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

# find initial solution
rootfinding.newtonRaphson(ebm)
T0 = ebm.T_avg()
print(f"Initial solution: T = {T0:.3f}")

# perform continuation
solutions = continuation.arclengthLoop(
    ebm,
    parameter_name,
    stepsize,
    tune_factor,
    parameter_range,
    maxContinuations,
)

# plot results
T_avgs = solutions["average"]
mus = solutions["parameter"]
stableBranch = solutions["stable"]
ax.plot(mus[stableBranch], T_avgs[stableBranch], "g.")
ax.plot(mus[~stableBranch], T_avgs[~stableBranch], "r--")

ax.set_ylabel(r"$\bar{T}$")
ax.set_xlabel(r"$\mu$")
ax.set_title(f"D = {ebm.D:.3f}")

plt.show()

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
stepsize = 0.1  # stepsize (needs to be small, why?)
tune_factor = 0.0005  # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations
initial_temperature = 200.0  # initial temperature

############### Continuation (first test) #################
ebm.T_coeffs[0] = initial_temperature
print("Performing continuation (once)...")
rootfinding = RootFinding(tolerance, maxiter)
rootfinding.output = False
rootfinding.newtonRaphson(ebm)
print(f"Initial solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")

errors = continuation.arclength(ebm, parameter_name, stepsize, tune_factor)
print(f"Continuation solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")

############### Continuation (loop) ##########################
print("Performing continuation (loop)...")
ebm.mu = parameter_range[0]
rootfinding = RootFinding(tolerance, maxiter)
ebm.T_coeffs = np.zeros(n_polys)
ebm.T_coeffs[0] = initial_temperature
rootfinding.output = False
rootfinding.newtonRaphson(ebm)

# TODO: perform continuation for a few initial temperatures
# initial_temperatures = np.linspace(200, 300, 5)
fig, ax = plt.subplots()

# for T0 in initial_temperatures:
#     ebm.T_coeffs = np.zeros(n_polys)
#     ebm.T_coeffs[0] = T0
#     # initial root
#     rootfinding.newtonRaphson(ebm)
#     T_avgs, mus, stableBranch = continuation.arclengthLoop(
#         ebm, parameter_name, stepsize, tune_factor, parameter_range, maxContinuations
#     )
#     mus = np.array(mus)
#     T_avgs = np.array(T_avgs)
#     stableBranch = np.array(stableBranch)

#     # plot results
#     ax.plot(mus[stableBranch], T_avgs[stableBranch], "g-")
#     ax.plot(mus[~stableBranch], T_avgs[~stableBranch], "r--")
ebm.T_coeffs = np.zeros(n_polys)
ebm.T_coeffs[0] = initial_temperature
T_avgs, mus, stableBranch = continuation.arclengthLoop(
    ebm, parameter_name, stepsize, tune_factor, parameter_range, maxContinuations
)
mus = np.array(mus)
T_avgs = np.array(T_avgs)
stableBranch = np.array(stableBranch)
plt.plot(mus[stableBranch], T_avgs[stableBranch], "g-")
plt.plot(mus[~stableBranch], T_avgs[~stableBranch], "r--")
ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$\bar{T}$")
ax.set_title("Continuation of the temperature profile")
plt.show()

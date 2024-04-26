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
continuation.output = False
parameter_name = "mu"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0, 100)  # range of the parametervalues of interest
stepsize = 0.1  # 0.1  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### Continuation (first test) #################
ebm.T_coeffs[0] = 220
print("Performing continuation (once)...")
rootfinding = RootFinding(tolerance, maxiter)
rootfinding.output = False
rootfinding.newtonRaphson(ebm)
print(f"Initial solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")

errors = continuation.arclength(ebm, parameter_name, stepsize, tune_factor)
print(f"Continuation solution:\n\tcoeffs = {ebm.T_coeffs}\n\taverage = {ebm.T_avg()}")

############### Continuation (loop) ##########################
print("Building bifurcation diagram (continuation loop)...")
D_values = [0.003, 0.3, 30]
stepsizes = [0.1, 0.1, 0.01]
maxContinuations_l = [1000, 1000, 10000]
tuning_factors = [0.00001, 0.00001, 0.00001]
fig, axs = plt.subplots(
    nrows=1, ncols=len(D_values), figsize=(8, 6), sharex=True, sharey=True
)
initial_temperatures = np.linspace(150, 600, 200)
for i, ax in enumerate(axs):
    ebm.D = D_values[i]
    # do some dirty hacks to get the continuation to go faster
    stepsize = stepsizes[i]
    maxContinuations = maxContinuations_l[i]
    tune_factor = tuning_factors[i]
    for T0 in initial_temperatures:
        print(f"Running continuation for D = {ebm.D:.3f} and T0 = {T0:1f}...")
        # initial guess
        ebm.mu = parameter_range[0]
        # if i == 2 and T0 > 280:
        #     ebm.mu = previousMu
        ebm.T_coeffs = np.zeros(n_polys)
        ebm.T_coeffs[0] = T0

        # find initial solution
        rootfinding.newtonRaphson(ebm)

        # perform continuation
        T_avgs, mus, stableBranch = continuation.arclengthLoop(
            ebm,
            parameter_name,
            stepsize,
            tune_factor,
            parameter_range,
            maxContinuations,
        )

        # plot results
        mus = np.array(mus)
        T_avgs = np.array(T_avgs)
        stableBranch = np.array(stableBranch)
        if np.any(T_avgs < 0):
            print("Negative temperatures encountered.")
            continue
        else:
            ax.plot(mus[stableBranch], T_avgs[stableBranch], "g-")
            ax.plot(mus[~stableBranch], T_avgs[~stableBranch], "r--")

        previousMu = ebm.mu

    # reset continuation
    continuation.previousSolution = None

    if i == 0:
        ax.set_ylabel(r"$\bar{T}$")
        ax.set_xlabel(r"$\mu$")

    ax.set_title(f"D = {ebm.D:.3f}")

# perform detailed search for D = 30
print("Performing detailed search for D = 30...")
ebm.D = 30
ebm.mu = 30
stepsize = 0.01
maxContinuations = 10000
tune_factor = 0.00001
initial_temperatures = np.linspace(280, 500, 200)
for T0 in initial_temperatures:
    print(f"Running continuation for D = {ebm.D:.3f} and T0 = {T0:1f}...")
    # initial guess
    ebm.T_coeffs = np.zeros(n_polys)
    ebm.T_coeffs[0] = T0

    # find initial solution
    rootfinding.newtonRaphson(ebm)

    # perform continuation
    T_avgs, mus, stableBranch = continuation.arclengthLoop(
        ebm,
        parameter_name,
        stepsize,
        tune_factor,
        parameter_range,
        maxContinuations,
    )

    # plot results
    mus = mus
    T_avgs = T_avgs
    stableBranch = stableBranch
    if np.any(T_avgs < 0):
        print("Negative temperatures encountered.")
        continue
    else:
        axs[2].plot(mus[stableBranch], T_avgs[stableBranch], "g-")
        axs[2].plot(mus[~stableBranch], T_avgs[~stableBranch], "r--")

axs[2].set_ylim(150, 350)

fig.suptitle("Continuation of the temperature profile")
fig.tight_layout()
fname = output_dir / "ebm_continuation.png"
plt.savefig(fname, bbox_inches="tight", dpi=300)

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils

pyutils.add_modules_to_path()
from continuation import Continuation
from mcm import MCM
from root_finding import RootFinding

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# set standard matplotlib style
pyutils.set_style()

############### Setup Non-linear System of Equations #####
mcm = MCM()
sys = mcm.constructSystem()
# mcm.printSystem()
# mcm.printDimensionlessSystem()
# mcm.findStationaryPoints()
stable_points = mcm.getStableStationaryPoints()
print("Stable stationary points:")
pprint(stable_points)
rootfinder = RootFinding()
rootfinder.output = True

############### Continuation parameters ##################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
parameter_name = "p1"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0.5, 1)  # range of the parametervalues of interest
stepsize = 0.01  # 0.01  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### Continuation (loop) ##########################
print("\nBuilding bifurcation diagram (continuation loop)...")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), sharex=True, sharey=True)

bifurcations = []
for i, stable_point in enumerate(stable_points):
    print(f"\n Continuation loop on {stable_point}")
    ax = axs[i]
    mcm.x = stable_point
    rootfinder.newtonRaphson(mcm)
    solutions = continuation.arclengthLoop(
        mcm,
        parameter_name,
        stepsize,
        tune_factor,
        parameter_range,
        maxContinuations,
    )
    averages = solutions["average"]
    minima = solutions["minimum"]
    maxima = solutions["maximum"]
    p1s = solutions["parameter"]
    stable = solutions["stable"]
    bifs = solutions["bifurcations"]

    # add new bifurcations to list
    bifurcations += bifs

    # plot bifurcations
    for bif in bifs:
        mean = np.mean(bif["solution"])
        ax.plot(bif["parameter"], mean, "ko", label="bifurcation")

    # stable branches
    # ax.plot(p1s[stable], maxima[stable], "r-", label="maximum")
    ax.plot(p1s[stable], averages[stable], "g-", label="average")
    # ax.plot(p1s[stable], minima[stable], "b-", label="minimum")

    # unstable branches
    unstable = ~stable
    # ax.plot(p1s[unstable], maxima[unstable], "r--")
    ax.plot(p1s[unstable], averages[unstable], "r--")
    # ax.plot(p1s[unstable], minima[unstable], "b--")

    if i == 0:
        ax.set_ylabel("$x$")
        ax.set_xlabel("$p_1$")
        ax.legend()

    ax.set_title(
        rf"$x_0  \approx ({stable_point[0]:.1f}, {stable_point[1]:.1f}, {stable_point[2]:.1f})$"
    )


############### Bifurcation points ##########################
print("\nBifurcation points:")
pprint(bifurcations)


fig.suptitle("Continuation of the stable stationary point(s)")
fig.tight_layout()
fname = output_dir / "mcm_continuation.png"
plt.savefig(fname, bbox_inches="tight", dpi=300)

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

############### Continuation parameters ##################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
parameter_name = "p1"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0.5, 1)  # range of the parametervalues of interest
stepsize = 0.01  # 0.1  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### Continuation (first test) #################
for stable_point in stable_points:
    print(f"\nTest continuation starting at {stable_point}")
    mcm.x = stable_point
    errors = continuation.arclength(mcm, parameter_name, stepsize, tune_factor)

############### Continuation (loop) ##########################
print("\nBuilding bifurcation diagram (continuation loop)...")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

for stable_point in stable_points:
    print(f"\n Continuation loop on {stable_point}")
    mcm.x = stable_point
    avgs, p1s, stableBranch = continuation.arclengthLoop(
        mcm,
        parameter_name,
        stepsize,
        tune_factor,
        parameter_range,
        maxContinuations,
    )

    ax.plot(p1s[stableBranch], avgs[stableBranch], "g-")
    ax.plot(p1s[~stableBranch], avgs[~stableBranch], "r--")

fig.suptitle("Continuation of the stable stationary point(s)")
fig.tight_layout()
fname = output_dir / "mcm_continuation.png"
plt.savefig(fname, bbox_inches="tight", dpi=300)

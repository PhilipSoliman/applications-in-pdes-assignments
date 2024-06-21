import json
from pprint import pprint

import latextable
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils
from scipy.integrate import solve_ivp

# from tabulate import tabulate
from texttable import Texttable

pyutils.add_modules_to_path()
from continuation import Continuation
from mcm import MCM
from root_finding import RootFinding

# figure output location
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# data folder
data_dir = root / "code" / "data"

# set standard matplotlib style
pyutils.set_style()

################ Setup Non-linear System of Equations #####
mcm = MCM()
n = 3  # number of variables


def fun(t: float, y: np.ndarray) -> np.ndarray:  # rhs of mcm for integration
    mcm.x = y
    f = mcm.evaluate()
    return f


############### setup continuation object ################
tolerance = 1e-5  # tolerance
maxiter = 100  # maximum number of iterations (per continuation)
continuation = Continuation(tolerance, maxiter)
continuation.output = True
continuation.parameterName = "p1"  # parameter to be continued (should correspond to an attribute of the EBM object)
parameter_range = (0.5, 1)  # range of the parametervalues of interest
stepsize = 0.001  # 0.01  # stepsize (needs to be small, why?)
tune_factor = 0.00001  # 0.001 # tune factor (needs to be small, why?)
maxContinuations = 1000  # maximum number of continuations

############### retrieve bifurcation points from data folder ################
with open(data_dir / "mcm_bifurcations.json", "r") as f:
    bifurcations = json.load(f)
    pprint(bifurcations)

x = bifurcations["solution"]
mcm.x = x
p1 = bifurcations["parameter"]
mcm.p1 = p1

print("\nFinding limit cycle starting from bifurcation point...")
cycle, cycle_valid = continuation.shootingMethod(
    mcm, "switch", period_guess=17.6, stepsize=0.0
)
cycle_point = cycle[:n]
h = cycle[n : 2 * n]
cycle_parameter = cycle[2 * n]
cycle_period = cycle[2 * n + 1]
# cycle_point = cycle[:n]
# cycle_parameter = cycle[n]
# cycle_period = cycle[n + 1]

solutions = []
parameters = []
periods = []
cycles = []
eigs = []
stable = []
it = 0
delta = 0.1
stable_tolerance = 1e-3
print("Cycle plot parameters: ")
num_cycles = 10 
print(f"\tnum_cycles: {num_cycles}")
stepsize = 0.005
print(f"\tstepsize: {stepsize}")
num_iterations = int((1 - cycle_parameter)//stepsize)
print(f"\tnum_iterations: {num_iterations}")  
cycle_spacing = num_iterations // num_cycles
print(f"\tcycle_spacing: {cycle_spacing}")
input("Press enter to generate cycle plot...")
while cycle_parameter <= 1 and it < num_iterations:

    it += 1
    if not cycle_valid:
        continue

    # calculate current stability
    monodromy = continuation.monodromyMatrix(
        mcm, cycle_point, cycle_parameter, cycle_period
    )
    eig = np.linalg.eigvals(monodromy)
    # check for approximate unity eigenvalues
    unityEigIndex = np.isclose(np.abs(eig), 1, atol=stable_tolerance)
    if np.all(np.abs(eig[~unityEigIndex]) < 1):
        stable.append(True)
    else:
        stable.append(False)
    eigs.append(eig.tolist())
    print(
        f"found cycle:"+
        f"\n\tT: {cycle_period:.3f}"+
        f"\n\tp: {cycle_parameter:.3f}"+
        f"\n\tx: ({cycle_point[0]:.2f}, {cycle_point[1]:.2f}, {cycle_point[2]:.2f})"+
        f"\n\teigs: l1={eig[0]:.2f}, l2={eig[1]:.2f}, l3={eig[2]:.2f}"
        f"\n\tstable: {stable[-1]}"
    )

    # store solution
    solutions.append(cycle_point.tolist())
    parameters.append(cycle_parameter)
    periods.append(cycle_period)

    # integrate over one period
    if it % cycle_spacing == 0:
        t_span = (0, cycle_period)
        t_eval = np.linspace(*t_span, 500)
        sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)
        cycles.append(sol.y.tolist())


    print(f"Continuing limit cycle with parameter {cycle_parameter}")
    mcm.x = cycle_point
    if it == 1:
        mcm.x += delta * h  # start with a small perturbation
    mcm.p1 = cycle_parameter
    cycle, cycle_valid = continuation.shootingMethod(
        mcm, "cont", period_guess=cycle_period, stepsize=stepsize
    )

    # set new cycle parameters
    cycle_point = cycle[:n]
    cycle_parameter = cycle[n]
    cycle_period = cycle[n + 1]


################## plot limit cycles ##################
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
num_cycles = len(cycles)
cycle_parameters = np.array(parameters)[::cycle_spacing]
parameter_range = np.max(cycle_parameters) - np.min(cycle_parameters)
tick_locs = (cycle_parameters-np.min(cycle_parameters))/parameter_range
colors = plt.cm.viridis(tick_locs)
for i, cycle in enumerate(cycles):
    cm = ax.scatter(cycle[0], cycle[1], cycle[2], "o", color=colors[i])

# set angle
ax.view_init(elev=20, azim=-120)

# set colorbar
scalarmappaple = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
cb = fig.colorbar(
    scalarmappaple,
    ax=ax,
    label="$p_1$",
)
cb.set_ticks(tick_locs[::2])
cycle_parameters_text = [f"{p:.2f}" for p in cycle_parameters[::2]]
cb.set_ticklabels(cycle_parameters_text)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

plt.show()

# save figure
filename = "mcm_limit_cycles"
filepath = output_dir / f"{filename}.png"
fig.savefig(filepath, bbox_inches="tight", dpi=500)


################### save limit cycles to file ##################
filename = "mcm_limit_cycles.json"
filepath = data_dir / filename
for bif in bifurcations:
    out = dict(
        points=solutions,
        parameter=parameters,
        periods=periods,
        cycles=cycles,
        # eigs=eigs, # complex numbers are not serializable
        stable=stable,
    )
    with open(filepath, "w") as f:
        json.dump(out, f, indent=4)

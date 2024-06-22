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


############## right hand side of the mcm ################
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

############### retrieve limit cycles ################
with open(data_dir / "mcm_limit_cycles.json", "r") as f:
    limit_cycles = json.load(f)
    solutions = limit_cycles["points"]
    if len(solutions) > 1:
        print("removing previous run")
        limit_cycles["points"] = [solutions[0]]
        limit_cycles["parameter"] = [limit_cycles["parameter"][0]]
        limit_cycles["periods"] = [limit_cycles["periods"][0]]
        limit_cycles["cycles"] = [limit_cycles["cycles"][0]]
        limit_cycles["stable"] = [limit_cycles["stable"][0]]
        limit_cycles["eigs"]["real"] = [limit_cycles["eigs"]["real"][0]]
        limit_cycles["eigs"]["imag"] = [limit_cycles["eigs"]["imag"][0]]


######## find first period doubling bifurcation point #####
def findPdoublingPoint(limit_cycles):
    """
    Finds period doubling point on the last limit cycle branch present in the limit_cycles dictionary.
    Uses interpolation to find the point, parameter and period at the period doubling point.
    """
    eigs = np.array(limit_cycles["eigs"]["real"][-1]) + 1j * np.array(
        limit_cycles["eigs"]["imag"][-1]
    )
    bifurcation_index = np.where(np.isclose(eigs + 1, 0, atol=1e-1))
    # eigs are decreasing to -1 so flip to get increasing order
    xp = np.flip(np.real(eigs)[bifurcation_index])
    points = np.array(limit_cycles["points"][-1])[bifurcation_index[0]]
    parameters = np.array(limit_cycles["parameter"][-1])[bifurcation_index[0]]
    periods = np.array(limit_cycles["periods"][-1])[bifurcation_index[0]]
    pdouble_point = np.zeros(3)
    for i in range(3):
        pdouble_point[i] = np.interp(-1, xp, np.flip(points[:, i]))
    pdouble_parameter = np.interp(-1, xp, np.flip(parameters))
    pdouble_period = np.interp(-1, xp, np.flip(periods))

    return pdouble_point, pdouble_parameter, pdouble_period


pdouble_point, pdouble_parameter, pdouble_period = findPdoublingPoint(limit_cycles)
t_span = (0, pdouble_period)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(fun, t_span, pdouble_point, t_eval=t_eval, vectorized=True)
pdouble_cycle = sol.y.tolist()
stable_tolerance = 1e-2
pdouble_stability = continuation.calculateLimitCycleStability(
    mcm, pdouble_point, pdouble_parameter, pdouble_period, tolerance=stable_tolerance
)
pdouble_eig = pdouble_stability[0]
print(
    f"Period doubling bifurcation:"
    + f"\n\tT: {pdouble_period:.3f}"
    + f"\n\tp: {pdouble_parameter:.3f}"
    + f"\n\tx: ({pdouble_point[0]:.2f}, {pdouble_point[1]:.2f}, {pdouble_point[2]:.2f})"
    + f"\n\teigs: l1={pdouble_eig[0]:.2f}, l2={pdouble_eig[1]:.2f}, l3={pdouble_eig[2]:.2f}"
    f"\n\tstable: {pdouble_stability[1]}"
)
############# finding limit cycles from period doubling point ############
print(
    "\nFinding limit cycle starting from period doubling point. Using initial conds.:"
    + f"\n\t2T: {2*pdouble_period:.2f}"
    + f"\n\tp1: {pdouble_parameter:.2f}"
    + f"\n\tx: ({pdouble_point[0]:.2f}, {pdouble_point[1]:.2f}, {pdouble_point[2]:.2f})"
)
mcm.x = pdouble_point
MCM.p1 = pdouble_parameter
cycle, cycle_valid = continuation.shootingMethod(
    mcm, "pdouble-switch", period_guess=2 * pdouble_period, stepsize=0.0
)
cycle_point = cycle[:n]
h = cycle[n : 2 * n]
cycle_parameter = cycle[2 * n]
cycle_period = cycle[2 * n + 1]
t_span = (0, cycle_period)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)

# calculate stability of period doubled limit cycle
stability = continuation.calculateLimitCycleStability(
    mcm, cycle_point, cycle_parameter, cycle_period, tolerance=stable_tolerance
)
eig = stability[0]
print(
    f"found period doubled cycle:"
    + f"\n\tT: {cycle_period:.3f}"
    + f"\n\tp: {cycle_parameter:.3f}"
    + f"\n\tx: ({cycle_point[0]:.2f}, {cycle_point[1]:.2f}, {cycle_point[2]:.2f})"
    + f"\n\teigs: l1={eig[0]:.2f}, l2={eig[1]:.2f}, l3={eig[2]:.2f}"
    f"\n\tstable: {stability[1]}"
)

solutions = [pdouble_point.tolist(), cycle_point.tolist()]
parameters = [pdouble_parameter, cycle_parameter]
periods = [pdouble_period, cycle_period]
cycles = [pdouble_cycle, sol.y.tolist()]
eigs = dict(
    real=[pdouble_eig.real.tolist(), eig.real.tolist()],
    imag=[pdouble_eig.imag.tolist(), eig.imag.tolist()],
)
stable = [pdouble_stability[1], stability[1]]

it = 0
delta = 0.01

print("Cycle plot parameters: ")
num_cycles = 10
print(f"\tnum_cycles: {num_cycles}")
stepsize = 0.001
print(f"\tstepsize: {stepsize}")
num_iterations = int((1 - cycle_parameter) // stepsize)
print(f"\tnum_iterations: {num_iterations}")
cycle_spacing = num_iterations // num_cycles
print(f"\tcycle_spacing: {cycle_spacing}")
input("Press enter to continue cycles and generate cycle plot...")
while cycle_parameter <= 1 and it < num_iterations:

    it += 1
    # if not cycle_valid:
    #     continue

    stability = continuation.calculateLimitCycleStability(
        mcm, cycle_point, cycle_parameter, cycle_period, tolerance=stable_tolerance
    )
    eig = stability[0]
    stable.append(stability[1])
    print(
        f"found cycle:"
        + f"\n\tT: {cycle_period:.3f}"
        + f"\n\tp: {cycle_parameter:.3f}"
        + f"\n\tx: ({cycle_point[0]:.2f}, {cycle_point[1]:.2f}, {cycle_point[2]:.2f})"
        + f"\n\teigs: l1={eig[0]:.2f}, l2={eig[1]:.2f}, l3={eig[2]:.2f}"
        f"\n\tstable: {stable[-1]}"
    )
    eigs["real"].append(eig.real.tolist())
    eigs["imag"].append(eig.imag.tolist())

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

################# plot limit cycles ##################
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
cycle_parameters = np.array(parameters)[
    np.linspace(1, len(parameters) - 1, num_cycles + 2).astype(int)
]
parameter_range = np.max(cycle_parameters) - np.min(cycle_parameters)
tick_locs = (cycle_parameters - np.min(cycle_parameters)) / parameter_range
colors = plt.cm.viridis(tick_locs)
for i in range(len(cycles) - 1):
    cycle = cycles[i + 1]
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
filename = "mcm_first_pdouble"
filepath = output_dir / f"{filename}.png"
fig.savefig(filepath, bbox_inches="tight", dpi=500)


################## save limit cycles to file ##################
filename = "mcm_limit_cycles.json"
filepath = data_dir / filename

limit_cycles["points"].append(solutions)
limit_cycles["parameter"].append(parameters)
limit_cycles["periods"].append(periods)
limit_cycles["cycles"].append(cycles)
limit_cycles["stable"].append(stable)
limit_cycles["eigs"]["real"].append(eigs["real"])
limit_cycles["eigs"]["imag"].append(eigs["imag"])

with open(filepath, "w") as f:
    json.dump(limit_cycles, f, indent=4)

# %%

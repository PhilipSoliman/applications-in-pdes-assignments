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

############## find limit cycle starting from bifurcation point ################
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
print("\nFinding limit cycle starting from bifurcation point...")
cycle = continuation.shootingMethod(mcm, period_guess=18)
print(cycle)

# extract point on limit cycle
cycle_point = cycle[:n,]
h_0 = cycle[n : 2 * n]
cycle_parameter = cycle[2 * n]
cycle_period = cycle[2 * n + 1]


# integrate over one period
# mcm.x = cycle_point
# mcm.p1 = cycle_parameter
# t_span = (0, 8 * cycle_period)
# t_eval = np.linspace(*t_span, 1000)

# sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)

# 3d plot of periodic solution using color gradient to rerpesent time

# k = sol.y.shape[1]
# colors = plt.cm.viridis(np.linspace(0, 1, k))
# ax.scatter(sol.y[0], sol.y[1], sol.y[2], "o", color=colors)

# mean = np.mean(sol.y)
# print(mean)

######################## find next limit cycle ########################
# print("\nFinding next limit cycle...")
# mcm.x = cycle_point + 0.1 * h_0
# mcm.p1 = cycle_parameter
# cycle = continuation.shootingMethod(mcm, period_guess=cycle_period)
# print(cycle)

# # extract point on limit cycle
# cycle_point = cycle[:n]
# h_0 = cycle[n : 2 * n]
# cycle_parameter = cycle[2 * n]
# cycle_period = cycle[2 * n + 1]
# mcm.p1 = cycle_parameter

# # integrate over one period
# t_span = (0, cycle_period)
# t_eval = np.linspace(*t_span, 1000)

# sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)

# k = sol.y.shape[1]
# colors = plt.cm.viridis(np.linspace(0, 1, k))
# ax.scatter(sol.y[0], sol.y[1], sol.y[2], "o", color=colors)

################## continue limit cycles ##################
# print("\nContinuing limit cycles...")
solutions = []
parameters = []
periods = []
cycles = []
while cycle_parameter <= 1:
    print(f"Continuing limit cycle with parameter {cycle_parameter}")
    # set initial guess for next limit cycle
    mcm.x = cycle_point + 0.13 * h_0
    mcm.p1 = cycle_parameter

    # find next limit cycle
    cycle = continuation.shootingMethod(mcm, period_guess=cycle_period)

    # extract point on limit cycle
    cycle_point = cycle[:n,]
    h_0 = cycle[n : 2 * n]
    cycle_parameter = cycle[2 * n]
    cycle_period = cycle[2 * n + 1]

    print(
        f"found new cycle with period {cycle_period} and parameter {cycle_parameter} at point {cycle_point}"
    )

    # store solution
    solutions.append(cycle_point)
    parameters.append(cycle_parameter)
    periods.append(cycle_period)

    # integrate over one period
    t_span = (0, cycle_period)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)
    cycles.append(sol.y)

################## plot limit cycles ##################
for sol in cycles:
    k = sol.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    ax.scatter(sol[0], sol[1], sol[2], "o", color=colors)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

plt.show()

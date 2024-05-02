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
print("\nFinding limit cycle starting from bifurcation point...")
cycle = continuation.findFirstLimitCycle(mcm)
print(cycle.y[:, -1])

# extract point on limit cycle
cycle_point = cycle.y[:n, -1]
h_0 = cycle.y[n : 2 * n, -1]
cycle_parameter = cycle.y[2 * n, -1]
cycle_period = cycle.y[2 * n + 1, -1]
mcm.p1 = cycle_parameter


# rhs of mcm
def fun(t: float, y: np.ndarray) -> np.ndarray:
    mcm.x = y
    f = mcm.evaluate()
    return f


# integrate over one period
t_span = (0, cycle_period)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)

# 3d plot of periodic solution using color gradient to rerpesent time
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
k = cycle.y.shape[1]
colors = plt.cm.viridis(np.linspace(0, 1, k))
ax.scatter(cycle.y[0], cycle.y[1], cycle.y[2], "o", color=colors)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

# plt.show()

######################## find next limit cycle ########################
print("\nFinding next limit cycle...")
mcm.x = cycle_point + 0.1 * h_0
mcm.p1 = cycle_parameter
cycle = continuation.findNextLimitCycle(mcm)
print(cycle.y[:, -1])

# extract point on limit cycle
cycle_point = cycle.y[:n, -1]
cycle_period = cycle.y[n, -1]
cycle_parameter = cycle.y[n + 1, -1]
mcm.p1 = cycle_parameter

# integrate over one period
t_span = (0, cycle_period)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(fun, t_span, cycle_point, t_eval=t_eval, vectorized=True)

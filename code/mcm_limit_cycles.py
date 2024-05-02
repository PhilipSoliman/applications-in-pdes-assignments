import json
from pprint import pprint

import latextable
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from helper import pyutils

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

############### setup continuation object ################
continuation = Continuation()
continuation.output = True

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


# find limit cycle using shooting method


# from scipy.integrate import solve_ivp


# def fun(t: float, y: np.ndarray) -> np.ndarray:
#     x = y[:3]
#     print(x)
#     mcm.x = x
#     f = mcm.evaluate()

#     phi = y[3:]
#     df = mcm.evaluate_derivative()

#     return np.append(f, df @ phi)

# t_span = (0, 1)

# y_0 = np.hstack((mcm.x, np.ones(3)))

# t_eval = np.linspace(*t_span, 1000)

# sol = solve_ivp(fun, t_span, y_0, t_eval=t_eval,vectorized=False)

# 3d plot of periodic solution
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(cycle.y[0], cycle.y[1], cycle.y[2])
# ax.set_xlabel("$x_1$")
# ax.set_ylabel("$x_2$")
# ax.set_zlabel("$x_3$")
# ax.set_title("Periodic solution")

# plt.show()

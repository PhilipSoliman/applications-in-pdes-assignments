#%%
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
from scipy.integrate import odeint
from tqdm import tqdm

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

################## load continuation data###############
with open(data_dir / "mcm_second_pdouble_backup.json", "r") as f:
    limit_cycles = json.load(f)
    point_pd3 = limit_cycles["points"][-1][0]
    parameter_pd1 = limit_cycles["parameter"][1][0]
    print("PD1:", parameter_pd1)
    parameter_pd2 = limit_cycles["parameter"][1][1]
    print("PD2:", parameter_pd2)
    parameter_pd3 = limit_cycles["parameter"][2][0]

############# feigenbaum's law ################
def feigenBaum(ln_m1, ln_m2) -> float:
    "returns the next parameter value for a period doubling bifurcation"
    feigenBaumConstant = 4.669201
    return (ln_m1 - ln_m2) / feigenBaumConstant + ln_m1

# predict next period doublings
ln_m2 = parameter_pd1
ln_m1 = parameter_pd2
parameters = [parameter_pd1, parameter_pd2]
for i in range(4):
    ln = feigenBaum(ln_m1, ln_m2)
    print(f"PD{i+3}:", ln)
    ln_m2 = ln_m1
    ln_m1 = ln
    parameters.append(ln)

############## right hand side of the mcm ################
def fun(y: np.ndarray, t) -> np.ndarray:  # rhs of mcm for integration
    mcm.x = y
    f = mcm.evaluate()
    return f

############## lyapunov exponents ################
def lyapunov_exponents(tmax, dt):
    Q = np.identity(n)
    exponents = np.zeros(n)

    t = np.linspace(0, tmax, int(tmax / dt))
    for i in tqdm(range(1, len(t))):
        mcm.x = odeint(fun, mcm.x, (t[i - 1], t[i]))[-1]
        # sol = solve_ivp(fun, (t[i - 1], t[i]),  mcm.x, method='RK45')
        # mcm.x = sol.y[:, -1]
        J = mcm.evaluate_derivative()
        Q = np.dot(J, Q) * dt + Q
        # Q = np.matmul( J * dt, Q)
        Q, R = np.linalg.qr(Q)

        exponents += np.log(np.abs(np.diag(R)))

    exponents /= tmax 
    return exponents


tmax = 100.0
dt = 0.001
print("start simulation")
exponents_l = []
for p in parameters:
    mcm.p1 = p
    mcm.x = point_pd3
    print("p_1:", p, end=" ")
    print()
    exponents = lyapunov_exponents(tmax, dt)
    print("Lyapunov exponents:", exponents)
    exponents_l.append(exponents)

#%% print latex table conttents
for i, parameter in enumerate(parameters):
    print(f"PD{i+1} & {parameter:.3f} & {exponents_l[i][0]:.2f}, {exponents_l[i][1]:.2f}, {exponents_l[i][2]:.2f} \\\\")

# %%

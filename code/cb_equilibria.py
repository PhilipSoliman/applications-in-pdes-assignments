"""
Model code for the Rac-Rho model
Y.M. Dijkstra, 2024
"""

import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import plottingtools as st
from Newton import Newton, Newton_small
from timeintegrator import time_integrator

colours = st.configure()
from copy import copy

### custom code for styling plots and saving figures
from helper import pyutils

pyutils.add_modules_to_path()
pyutils.set_style()
root = pyutils.get_root()
output_dir = root / "report" / "figures"

"""
########################################################################################################################
Functions and Jacobians
########################################################################################################################
"""


def Fun(y, B, params):  ## Functions for Rac and Rho, returns the d/dt
    A, R0, rho0, deltaR, n = params
    R = y[0]
    rho = y[1]
    F = np.zeros(2)
    F[0] = A / (rho0**n + rho**n) * (1 - R) - deltaR * R
    F[1] = B / (R0**n + R**n) * (1 - rho) - rho
    return F


def Fun_ext(y, params):  ## Extended function with parameter; with F[2] dB/dt.
    A, R0, rho0, deltaR, n, Bmax, eps = params
    F = np.zeros(3)
    B = y[2]
    F[:2] = Fun(y[:2], B, (A, R0, rho0, deltaR, n))
    ##### adapt the code here for changing dB/dt
    F[2] = eps
    # F[2] = eps*(Bmax-B)/Bmax
    #####
    return F


def Jac(y, B, params):  ## Jacobian
    A, R0, rho0, deltaR, n = params
    R = y[0]
    rho = y[1]
    J = np.zeros((2, 2))
    J[0, 0] = -A / (rho0**n + rho**n) - deltaR
    J[0, 1] = -A * (1 - R) * n * rho ** (n - 1) / (rho0**n + rho**n) ** 2
    J[1, 0] = -B * (1 - rho) * n * R ** (n - 1) / (R0**n + R**n) ** 2
    J[1, 1] = -B / (R0**n + R**n) - 1
    return J


def Jac_B(y, B, params):  ## Derivative w.r.t. B
    R = y[0]
    rho = y[1]
    JB = np.zeros(2)
    JB[1] = (1 - rho) / (R0**n + R**n)
    return JB


"""
########################################################################################################################
Commands for running the scripts
########################################################################################################################
"""
########################################################################################################################
# Initialise parameters
########################################################################################################################
A = 0.003
R0 = 0.3
rho0 = 0.16
deltaR = 1
n = 4
ds = 0.001
B0 = 0.03
Bmax_bifdiag = 0.1  # maximum value of B for continuation; terminate for larger B

eps = 0.0  # rate of change of B, used in exercise 3, 4
Bmax = 0.04  # value Bmax, only used in exercise 4

dt = 0.01
T = 20


# initialise arrays for output
params = (A, R0, rho0, deltaR, n)
params_ext = (A, R0, rho0, deltaR, n, Bmax, eps)
B_arr = []  # solution y
y_arr = []  # parameter B
evs = []  # largest eigenvalues
all_evs = []  # list of all eigenvalues
bifurcations = []  # list of bifurcation points

########################################################################################################################
## Find equilibria for variable B=B0
########################################################################################################################
initial_guesses = [[0.8,0], [0.5, 0.2], [0, 0.8]]
roots = []
norms = []
for i, y0 in enumerate(initial_guesses):
    B = B0
    y = Newton_small(Fun, Jac, y0, B0, params)
    roots.append(y)
    norm = np.linalg.norm(y)
    norms.append(norm)

# sort the roots by the norm
roots = [x for _, x in sorted(zip(norms, roots))]
norms.sort()
print(r"\begin{align*}")
for i, y in enumerate(roots):
    norm = norms[i]
    print(r"E_" + f"{i}" + r"=(\hat{R_a}, \hat{\rho_a}) \approx (" + f"{y[0]:.2f}," + f"{y[1]:.2f}" + r") \leftarrow ||E_" + f"{i}" +  r"|| \approx " + f"{norm:.2f}", end="")
    if i < len(roots) - 1:
        print(r", \\ ")
    else:
        print(r".")    
print(r"\end{align*}")

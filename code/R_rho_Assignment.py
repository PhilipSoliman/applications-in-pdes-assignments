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
B0 = 0.001
Bmax_bifdiag = 0.1  # maximum value of B for continuation; terminate for larger B

eps = 0.001   # rate of change of B, used in exercise 3, 4
Bmax = 0.04  # value Bmax, only used in exercise 4

dt = 0.01
T = 100


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
y0 = np.array([1, 1])
B = B0
y = Newton_small(Fun, Jac, y0, B, params)

########################################################################################################################
# continuation
########################################################################################################################
# add results for B0 to the array
B_arr.append(B)
y_arr.append(y)
J_ev = Jac(y, B, params)
ev = np.linalg.eigvals(J_ev)
evs.append(np.max(np.real(ev)))
all_evs.append((np.real(ev)))

# start continuation
iter = 0
while B < Bmax_bifdiag and iter < 3000:
    # predictor
    J = np.zeros((3, 3))
    rhs = np.zeros(3)
    J[:2, :2] = Jac(y, B, params)
    J[:2, -1] = Jac_B(y, B, params)
    J[-1, 1] = 1
    rhs[-1] = 1
    z = np.linalg.solve(J, rhs)  # z contains components da/ds and dmu/ds
    z_rescale = z / np.sqrt(np.sum(z**2))  # now scaled s.t. total step length is ds
    y_pred = y + z_rescale[:-1] * ds
    B_pred = B + z_rescale[-1] * ds

    # corrector
    y, B = Newton(Fun, Jac, Jac_B, y_pred, B_pred, y, B, ds, params)

    # compute eigenvalues for assessing stability
    J_ev = Jac(y, B, params)
    ev = np.linalg.eigvals(J_ev)
    real_ev = np.real(ev)
    evs.append(np.max(np.real(ev)))  # save maximum real value of evs
    if iter >= 2 and np.sign(evs[-1]) != np.sign([evs[-2]]):
        bifurcations.append((y,B))
    all_evs.append((np.real(ev)))

    # update the output arrays
    B_arr.append(B)
    y_arr.append(y)
    iter += 1

# print nicely formatted biurcation points
for bif in bifurcations:
    print(f"Bifurcation at B={bif[1]:.3f}, Rac={bif[0][0]:.3f}, rho={bif[0][1]:.3f}")

# convert lists to arrays
B_arr = np.asarray(B_arr)  # array of values of B
y_arr = np.asarray(
    y_arr
)  # array with values of y, shape (length(B), 2), where the 2nd dimension has entries Rac, rho
evs = np.asarray(evs)
all_evs = np.asarray(all_evs)
y_s = copy(y_arr)  # split y into stable ('y_s') and unstable ('y_u')
y_s[np.where(evs > 0)[0], :] = np.nan
y_u = copy(y_arr)
y_u[np.where(evs < 0)[0], :] = np.nan
evs_s = copy(evs)
evs_s[np.where(evs > 0)] = np.nan
evs_u = copy(evs)
evs_u[np.where(evs < 0)] = np.nan
all_evs_s = copy(all_evs)
all_evs_s[np.where(evs > 0)[0], :] = np.nan
all_evs_u = copy(all_evs)
all_evs_u[np.where(evs < 0)[0], :] = np.nan

########################################################################################################################
# Time integrator
########################################################################################################################
# parameter lists
params_ext = (A, R0, rho0, deltaR, n, Bmax, eps)

# the time integrator has arguments: function, parameter list (see above), tuple of initial conditions (R, rho, B), time step, end time.
# y_tim has shape (3, length(t)). In the first dimension it has entries Rac, rho, B
ic = (0.1, 0.5, B0)
t, y_tim = time_integrator(Fun_ext, params_ext, ic, dt, T)

########################################################################################################################
# Plot
########################################################################################################################
fig = plt.figure(1, figsize=(2, 2))
plt.subplot(2, 2, 1)

# Plot result of continuation Rac vs B
plt.plot(B_arr, y_s[:, 0], "-", color=colours[0])
plt.plot(B_arr, y_u[:, 0], "--", color=colours[0])

## Plot trajectory in same plot
plt.plot(
    y_tim[-1, :], y_tim[0, :], "-", color=colours[1]
)  # trajectory y_tim[-1,:] = B(t), y_tim[0,:] = Rac(t)
plt.plot(y_tim[-1, 0], y_tim[0, 0], "o", color=colours[1])  # mark initial condition
plt.xlabel(r"$\hat{B}$")
plt.ylabel(r"$\hat{R}$")
plt.ylim(0, 1)
plt.xlim(0, Bmax_bifdiag)

## plot bifurcation points
for bif in bifurcations:
    plt.plot(bif[1], bif[0][0], "o", color=colours[2])

plt.subplot(2, 2, 2)
# Plot result of continuation rho vs B
plt.plot(B_arr, y_s[:, 1], "-", color=colours[0])
plt.plot(B_arr, y_u[:, 1], "--", color=colours[0])

## Plot trajectory in same plot
plt.plot(
    y_tim[-1, :], y_tim[1, :], "-", color=colours[1]
)  # trajectory y_tim[-1,:] = B(t), y_tim[0,:] = rho(t)
plt.plot(y_tim[-1, 0], y_tim[1, 0], "o", color=colours[1])  # mark initial condition
plt.xlabel(r"$\hat{B}$")
plt.ylabel(r"$\hat{\rho}$")
plt.ylim(0, 1)
plt.xlim(0, Bmax_bifdiag)

## plot bifurcation points
for bif in bifurcations:
    plt.plot(bif[1], bif[0][1], "o", color=colours[2])

plt.subplot(2, 2, 3)
## Plot result of continuation Rac vs rho
plt.plot(y_s[:, 0], y_s[:, 1], "-", color=colours[0])
plt.plot(y_u[:, 0], y_u[:, 1], "--", color=colours[0])

## Plot trajectory in same plot
plt.plot(y_tim[0, :], y_tim[1, :], "-", color=colours[1])
plt.plot(y_tim[0, 0], y_tim[1, 0], "o", color=colours[1])
Bplot = 1
plt.xlabel(r"$\hat{R}$")
plt.ylabel(r"$\hat{\rho}$")
plt.ylim(0, 1)
plt.xlim(0, 1)

## plot bifurcation points
for bif in bifurcations:
    plt.plot(bif[0][0], bif[0][1], "o", color=colours[2])

plt.subplot(2, 2, 4)
# B as function of t
plt.plot(t, y_tim[2, :], "-", color=colours[1])
plt.ylabel(r"$\hat{B}$")
plt.xlabel(r"$\hat{t}$")

st.show()

# save figure
filename = (
    f"cb_R(0)={ic[0]}_rho(0)={ic[1]}_B(0)_{ic[2]}_eps={eps}_Bmax={Bmax}.png"
)
fig.savefig(output_dir / filename)

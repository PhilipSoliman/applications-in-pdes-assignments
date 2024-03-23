import numpy as np
import matplotlib.pyplot as plt
from helper import pyutils

pyutils.add_modules_to_path()
from ebm import EBM
from root_finding import RootFinding

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n_polys = 5  # number of Legendre polynomials
n_quads = 2 * n_polys  # number of quadrature points
grid_resolution = 100  # resolution of the grid

############### Setup Non-linear System of Equations #####
ebm = EBM(n_polys, n_quads, grid_resolution)

############### Generating initial guesses for T #########
initial_temperatures = np.linspace(150, 350, 100)

############### find fixed points (using NR and Broyden) #
rootfinding = RootFinding(maxiter=1000)
rootfinding.output = False
fig, [ax_nr, ax_br] = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)

# Newton-Raphson
fixed_points = np.zeros(len(initial_temperatures))
iterations = np.zeros(len(initial_temperatures))
for i, initial_temperature in enumerate(initial_temperatures):
    ebm.T_coeffs = np.zeros(n_polys)
    ebm.T_coeffs[0] = initial_temperature
    errors = rootfinding.newtonRaphson(ebm)
    if rootfinding.converged:
        fixed_points[i] = ebm.T_avg()
        if ebm.T_avg() < 0:
            fixed_points[i] = np.nan
            print("negative temperature encountered")
    else:
        fixed_points[i] = np.nan
    iterations[i] = len(errors)

ax_nr.scatter(
    initial_temperatures,
    fixed_points,
    c=(np.max(iterations) - iterations) ** 2,
    cmap="Oranges_r",
    s=100,
    marker="o",
    alpha=0.8,
)
ax_nr.set_xlabel("$T_0$")
ax_nr.set_ylabel(r"$\bar{T}$")
ax_nr.set_title("NR")

# Broyden's method
fixed_points = np.zeros(len(initial_temperatures))
iterations = np.zeros(len(initial_temperatures))
for i, initial_temperature in enumerate(initial_temperatures):
    ebm.T_coeffs = np.zeros(n_polys)
    ebm.T_coeffs[0] = initial_temperature
    errors = rootfinding.broydensMethod(ebm)
    if rootfinding.converged:
        fixed_points[i] = ebm.T_avg()
    else:
        fixed_points[i] = np.nan
    iterations[i] = len(errors)

# plot fixed points
ax_br.scatter(
    initial_temperatures,
    fixed_points,
    c=(np.max(iterations) - iterations) ** 2,
    cmap="Oranges_r",
    s=100,
    marker="o",
    alpha=0.8,
)
ax_br.set_title("Broydens")
fig.tight_layout()
fig.suptitle("Fixed points of the EBM", fontsize=16)
fn = output_dir / "fixed_points.png"
fig.savefig(fn, dpi=500)

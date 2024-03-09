import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from custom_pyutils import pyutils
from typing import Callable
from timeit import timeit
from non_linear_system import NonLinearSystem

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n = 10  # number of Legendre polynomials
number_of_quad_points = 2 * n # number of quadrature points
grid_resolution = 100  # resolution of the grid
tolerance = 1e-6  # tolerance for Newton-Raphson method

############### Setup Non-linear System of Equations ####################
NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

print("Problem constants:\n  ", end="")
pprint({k: v for k, v in NLS.__dict__.items()})

print("Plotting initial guess for T...")
T_initial = NLS.T_x(NLS.x)
plt.close()
plt.plot(NLS.x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)
print("Saved initial guess for T to:\n  ", fn)

################### Newton-Raphson Method ####################

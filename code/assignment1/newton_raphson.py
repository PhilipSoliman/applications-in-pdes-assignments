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
number_of_quad_points = (
    2 * n
)  # number of quadrature points for Gauss quadrature (since we are multiplying polynomial of degree n with another polynomial of degree n, we need 2n quadrature points for exact integration)
grid_resolution = 100  # resolution of the grid
tolerance = 1e-6  # tolerance for Newton-Raphson method

############### Setup Non-linear System of Equations ####################
NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

print("Problem constants:\n  ", end="")
pprint(
    {k: v for k, v in NLS.__dict__.items()}
) 

############### Setup Numerical Integration ####################
# Legendre polynomials and eigenvalues
legendre_polys, legendre_eigs = NLS.legendre_polys_callable, NLS.legendre_eigs
for i in range(n):
    plt.plot(NLS.x, legendre_polys[i](NLS.x), label=f"$\phi_{i}$")
plt.legend(fontsize=8)
plt.title(f"First {n} Legendre Polynomials")
fn = output_dir / f"legendre_polynomials_n={n}.png"
plt.savefig(fn, dpi=500)

############### Setup Non-linear System of Equations ####################
T_initial = NLS.T_x(NLS.x)
plt.close()
plt.plot(NLS.x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)

print(
    "Shape of F(T): \n  ",
    NLS.evaluate().shape,
)
print(
    "Shape of F_T(T): \n  ",
    NLS.evaluate_derivative().shape,
)

################## timing function evaluations ##################
timing_loops = 1000
F_time = timeit(
    "NLS.evaluate()",
    globals=globals(),
    number=timing_loops,
)

F_T_time = timeit(
    "NLS.evaluate_derivative()",
    globals=globals(),
    number=timing_loops,
)

print(f"Time to evaluate F(T) (averaged over {timing_loops} runs): {F_time/1000:.2e} s")
print(f"Time to evaluate F_T(T) (averaged over {timing_loops} runs): {F_T_time/1000:.2e} s")



################### Newton-Raphson Method ####################

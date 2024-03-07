import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from custum_pyutils import pyutils
from typing import Callable
# import problem functions and constants
from definitions import *

# print all constants and values
print("Problem constants:\n  ", end="")
pprint({k: v for k, v in globals().items() if isinstance(v, (int, float))}) # Callable))})

# set figure output directory
root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

# set standard matplotlib style
pyutils.set_style()

############### Computation parameters ####################
n = 10  # number of Legendre polynomials
number_of_quad_points = 2*n  # number of quadrature points for Gauss quadrature
grid_resolution = 100  # resolution of the grid
x = np.linspace(-1, 1, grid_resolution)  # grid points


############### Setup Numerical Integration ####################
# Legendre polynomials and eigenvalues
legendre_polys, legendre_eigs = generate_legendre_polynomials(n)
for i in range(n):
    plt.plot(x, legendre_polys[i](x), label=f"$\phi_{i}$")
plt.legend(fontsize=8)
plt.title(f"First {n} Legendre Polynomials")
fn = output_dir / f"legendre_polynomials_n={n}.png"
plt.savefig(fn, dpi=500)
print("lamda_n: \n  ", end=" ")
pprint(legendre_eigs)

# numerical integration
sample_points, quad_weights = get_leggauss_quadrature(number_of_quad_points)
print("Gauss quadrature weights: \n  ", quad_weights)
print("Gauss quadrature points: \n  ", sample_points)

# evaluate legendre polynomials at sample points
legendre_polys_at_sample_points = np.array([poly(sample_points) for poly in legendre_polys])


############### Setup Non-linear System of Equations ####################
# initial guess for T
T_coeffs = np.zeros(n)
T_initial = T_x(x, T_coeffs, legendre_polys)
plt.close()
plt.plot(x, T_initial, label="Initial guess")
plt.title("Initial guess for T")
plt.legend(fontsize=8)
fn = output_dir / "initial_guess_T.png"
plt.savefig(fn, dpi=500)

print("Shape of F(T): \n  ", F(T_coeffs, sample_points, quad_weights, legendre_polys_at_sample_points, legendre_eigs).shape)
if n <= 10: print("F(T): \n  ", F(T_coeffs, sample_points, quad_weights, legendre_polys_at_sample_points, legendre_eigs))
print("Shape of F_T(T): \n  ", F_T(T_coeffs, sample_points, quad_weights, legendre_polys_at_sample_points).shape)
if n <= 10: print("F_T(T): \n  ", F_T(T_coeffs, sample_points, quad_weights, legendre_polys_at_sample_points))



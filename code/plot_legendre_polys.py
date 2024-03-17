import matplotlib.pyplot as plt
from helper import pyutils
from ebm import NonLinearSystem
import numpy as np

root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

pyutils.set_style()

n = 10  # number of Legendre polynomials
number_of_quad_points = 4 * n # number of quadrature points
grid_resolution = 100  # resolution of the grid

NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

for poly in NLS.legendre_polys_callable:
    plt.plot(NLS.x, poly(NLS.x), label=f"$\phi_{NLS.legendre_polys_callable.index(poly)}$")
plt.title("Legendre Polynomials")
plt.xlabel("x")
plt.ylabel("$\phi_i(x)$")
plt.legend(fontsize=8)
fn = output_dir / f"legendre_polynomials_n={n}.png"
plt.savefig(fn, dpi=500)
print("Saved legendre polynomials to:\n  ", fn)

# check orthogonality of the Legendre polynomials
PtP = np.einsum("is,js->ijs", NLS.legendre_polys, NLS.legendre_polys)
PtP = PtP @ NLS.quad_weights
print(PtP)
print(PtP.diagonal())
print(NLS.legendre_norms)
PtP_significant = np.zeros_like(PtP)
PtP_significant[PtP>1.0e-12] = PtP[PtP>1.0e-12]
print(PtP_significant)

# check value of jacobian
F_T = NLS.evaluate_derivative()
print(F_T)
F_T_significant = np.zeros_like(F_T)
F_T_significant[F_T>1.0e-12] = F_T[F_T>1.0e-12]
print(F_T_significant)
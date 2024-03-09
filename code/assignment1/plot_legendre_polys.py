import matplotlib.pyplot as plt
from custom_pyutils import pyutils
from non_linear_system import NonLinearSystem

root = pyutils.get_root()
output_dir = root / "report" / "figures"
print(f"Output directory:\n  {output_dir}")

pyutils.set_style()

n = 10  # number of Legendre polynomials
number_of_quad_points = 2 * n # number of quadrature points
grid_resolution = 100  # resolution of the grid

NLS = NonLinearSystem(n, number_of_quad_points, grid_resolution)

legendre_polys, eigs = NLS.generate_legendre_polynomials()
for poly in legendre_polys:
    plt.plot(NLS.x, poly(NLS.x), label=f"$\phi_{legendre_polys.index(poly)}$")
plt.title("Legendre Polynomials")
plt.xlabel("x")
plt.ylabel("$\phi_i(x)$")
plt.legend(fontsize=8)
fn = output_dir / f"legendre_polynomials_n={n}.png"
plt.savefig(fn, dpi=500)
print("Saved legendre polynomials to:\n  ", fn)

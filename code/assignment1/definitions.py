import numpy as np
from scipy.special import legendre
from typing import Callable

# easy guassian quadrature integration
from scipy.integrate import fixed_quad as guassian_quadrature_integration

# gauss-legendre quadrature weights (more efficient as no for loops are used)
from numpy.polynomial.legendre import leggauss

# problem constants and parameters (capitalized is default value)
MU = 30.0  # greenhouse gas and particle parameter
a_1 = 0.278  # albedo of water
a_2 = 0.7  # albedo of ice
T_star = 273.15  # freezing temperature
delta = 0  # heat flux at poles
Q_0 = 341.3  # solar radiation constant
sigma_0 = 5.67e-8  # Stefan-Boltzmann constant
epsilon_0 = 0.61  # emissivity of the Earth
C_T = 5.0e8  # heat capacity of the Earth


# solar radiation (R_A)
def R_A(x: np.ndarray, T: np.ndarray, mu: float = MU) -> np.ndarray:
    return Q_solar(x) * (1 - albedo(T)) + mu


def dR_A(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    return Q_solar(x) * dalbedo(T)


def Q_solar(x: np.ndarray) -> np.ndarray:
    return Q_0 * (1 - 0.241 * (1 - x**2))


def albedo(T: np.ndarray) -> np.ndarray:
    return a_1 + (a_2 - a_1) / 2 * (1 + np.tanh(T - T_star))


def dalbedo(T: np.ndarray) -> np.ndarray:
    return (a_2 - a_1) / 2 * 1 / np.cos(T - T_star) ** 2


# Black body radiation (R_E)
def R_E(T: np.ndarray) -> np.ndarray:
    return epsilon_0 * sigma_0 * T**4


def dR_E(T: np.ndarray) -> np.ndarray:
    return 4 * epsilon_0 * sigma_0 * T**3


# dispersion & legendre polynomials
def R_D(
    T_coeffs: np.ndarray, legendre_polys: np.ndarray, legendre_eigs: np.ndarray
) -> np.ndarray:
    """
    Evaluate the dispersion term using the coefficients and eigenvalues
    of the respective Legendre polynomials.

    Returns:
    --------
    np.ndarray (m,): The dispersion term evaluated at the given coefficients.
        m is trhe number of sample points.
    """
    return (legendre_eigs * T_coeffs) @ legendre_polys


def generate_legendre_polynomials(n: int) -> tuple[list, np.ndarray]:
    polys = [legendre(i) for i in range(n)]
    eigs = np.array([-i * (i + 1) for i in range(n)])
    return (polys, eigs)


# evaulate temperature
def T(T_coeffs: np.ndarray, legendre_polys: np.ndarray) -> np.ndarray:
    return T_coeffs @ legendre_polys  # efficient; used for numerical integration


def T_x(
    x: np.ndarray,
    T_coeffs: np.ndarray,
    legendre_polys: list[Callable],
) -> np.ndarray:
    legendre_polys = np.array([poly(x) for poly in legendre_polys])
    return T(T_coeffs, legendre_polys)  # not efficient;  used for visualisation


# Gauss quadrature weights and points
def get_leggauss_quadrature(n: int) -> tuple[list[np.ndarray]]:
    return leggauss(n)

# full energy balance and its derivative w.r.t. T (coefficients of legendre polys)
def energy_balance(
    x: np.ndarray,
    T_coeffs: np.ndarray,
    legendre_polys: np.ndarray,
    legendre_eigs: np.ndarray,
) -> np.ndarray:
    T_eval = T(T_coeffs, legendre_polys)
    return R_D(T_coeffs, legendre_polys, legendre_eigs) + R_A(x, T_eval) - R_E(T_eval)


def energy_balance_derivative(
    x: np.array,
    T_coeffs: np.ndarray,
    legendre_polys: np.ndarray,
    legendre_eigs: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the derivative of the energy balance terms w.r.t. to (the coefficients of) T
    using the coefficients and eigenvalues of the respective Legendre polynomials.

    The derivative of the dispersion term are the eigenvalues of the Laplacian
    multiplied by the Legendre polynomials. The derivative of the solar radiation
    and the black body radiation are their derivative w.r.t. T multiplied by the
    Legendre polynomials (chain rule).

    Returns:
    --------
    np.ndarray (n, m): The derivative of the energy balance terms evaluated at the given coefficients.
        m is the number of sample points and n is the number of Legendre polynomials.
    """
    T_eval = T(T_coeffs, legendre_polys)
    return (
        legendre_eigs[:, None] * legendre_polys
        + (dR_A(x, T_eval) - dR_E(T_eval)) * legendre_polys
    )


# non-linear system of equations
def F(
    T_coeffs: np.ndarray,
    quad_samples: np.ndarray,
    quad_weights: np.ndarray,
    legendre_polys: np.ndarray,
    legendre_eigs: np.ndarray,
) -> np.ndarray:
    integrand = energy_balance(quad_samples, T_coeffs, legendre_polys, legendre_eigs)
    test_function = legendre_polys
    weak_form = integrand * test_function
    return weak_form @ quad_weights[:, None]


def F_T(
    T_coeffs: np.ndarray,
    quad_samples: np.ndarray,
    quad_weights: np.ndarray,
    legendre_polys: np.ndarray,
    legendre_eigs: np.ndarray,
) -> np.ndarray:
    integrand = energy_balance_derivative(
        quad_samples, T_coeffs, legendre_polys, legendre_eigs
    )
    test_function = legendre_polys
    weak_form_derivative = np.einsum(
        "is,js->ijs", test_function, integrand
    )  # tensor such that out[i,j] = (dR_A(T) - dR_E(T)) * phi_j * phi_i
    return np.einsum("ijs,s->ij", weak_form_derivative, quad_weights)

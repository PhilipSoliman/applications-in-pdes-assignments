import numpy as np
from scipy.special import legendre
from typing import Callable

# easy guassian quadrature integration
from scipy.integrate import fixed_quad as guassian_quadrature_integration

# from definitions import *

# gauss-legendre quadrature weights (more efficient as no for loops are used)
from numpy.polynomial.legendre import leggauss


class NonLinearSystem:

    # problem constants and parameters (capitalized is default value)
    mu_default = 30.0  # greenhouse gas and particle parameter
    a_1 = 0.278  # albedo of water
    a_2 = 0.7  # albedo of ice
    T_star = 273.15  # freezing temperature
    delta = 0  # heat flux at poles
    Q_0 = 341.3  # solar radiation constant
    sigma_0 = 5.67e-8  # Stefan-Boltzmann constant
    epsilon_0 = 0.61  # emissivity of the Earth
    C_T = 5.0e8  # heat capacity of the Earth
    D_default = 0.3 # heat dispersion coefficient

    def __init__(self, n_polys: int, n_quad_points: int, n_gridpoints: int) -> None:
        self.n_polys = n_polys
        self.n_quad_points = n_quad_points
        self.n_gridpoints = n_gridpoints
        self.legendre_polys_callable, self.legendre_eigs = (
            self.generate_legendre_polynomials()
        )
        self.quad_samples, self.quad_weights = self.get_leggauss_quadrature()
        # evaluate legendre polynomials at sample points
        self.legendre_polys = np.array(
            [poly(self.quad_samples) for poly in self.legendre_polys_callable]
        )

        # setup grid points
        self.x = np.linspace(-1, 1, n_gridpoints)

        # default values of properties
        self._T_coeffs = np.zeros(n_polys)
        self._mu = NonLinearSystem.mu_default
        self._D = NonLinearSystem.D_default

    # legendre polynomials and eigenvalues
    def generate_legendre_polynomials(self) -> tuple[list, np.ndarray]:
        polys = [legendre(i) for i in range(self.n_polys)]
        eigs = np.array([-i * (i + 1) for i in range(self.n_polys)])
        return (polys, eigs)

    # Gauss quadrature weights and points
    def get_leggauss_quadrature(self) -> tuple[list[np.ndarray]]:
        return leggauss(self.n_quad_points)

    # proqerties
    @property
    def T_coeffs(self) -> np.ndarray:
        return self._T_coeffs

    @T_coeffs.setter
    def T_coeffs(self, T_coeffs: np.ndarray) -> None:
        self._T_coeffs = T_coeffs

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, mu: float) -> None:
        self._mu = mu

    @property
    def D(self) -> float:
        return self._D

    @D.setter
    def D(self, D: float) -> None:
        self._D = D

    def T_eval(self) -> np.ndarray:
        """
        Vectorised evaluation at quadrature sample points. Efficient. Used for numerical integration
        """
        return self.T_coeffs @ self.legendre_polys

    def T_x(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the temperature at the given grid points (x) using the T_coeffs.
        Not efficient. Used for visualisation.
        """
        legendre_polys_eval = np.array(
            [poly(x) for poly in self.legendre_polys_callable]
        )
        return self.T_coeffs @ legendre_polys_eval

    ######## non-linear system of equations ########
    def evaluate(self) -> np.ndarray:
        """
        perfroms numerical integration using Gauss quadrature to evaluate the non-linear system of equations.
        for a given temperature (T_coeffs), the energy balance and Legendre polys are evaluated at the
        quadrature sample points.

        Returns:
        --------
        np.ndarray: vector such that out[i] = int_{-1}{1} [(R_D(T) + R_A(T) - R_E(T))] * phi_i dx

        """
        integrand = self.energy_balance(self.quad_samples)
        test_function = self.legendre_polys
        weak_form = integrand * test_function
        return weak_form @ self.quad_weights[:, None]

    def evaluate_derivative(self) -> np.ndarray:
        """

        Returns:
        --------
        np.ndarray: matrix such that out[i,j] = int_{-1}{1} [(lambda_j + dR_A(T) - dR_E(T)) * phi_j] * phi_i dx
        """
        integrand = self.energy_balance_derivative(self.quad_samples)
        test_function = self.legendre_polys
        weak_form_derivative = np.einsum("is,js->ijs", test_function, integrand)
        return np.einsum("ijs,s->ij", weak_form_derivative, self.quad_weights)

    ######## Energy balance terms ########
    def energy_balance(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        T_eval = self.T_eval()  # perform evaluate here not in the functions
        # to avoid redundant computation
        return self.R_D() + self.R_A(x, T_eval) - self.R_E(T_eval)

    def energy_balance_derivative(
        self,
        x: np.array,
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
        T_eval = self.T_eval()
        return (
            self.legendre_eigs[:, None] * self.legendre_polys
            + (self.dR_A(x, T_eval) - self.dR_E(T_eval)) * self.legendre_polys
        )

    # solar radiation (R_A)
    def R_A(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        return NonLinearSystem.Q_solar(x) * (1 - NonLinearSystem.albedo(T)) + self.mu

    @staticmethod
    def dR_A(x: np.ndarray, T: np.ndarray) -> np.ndarray:
        return NonLinearSystem.Q_solar(x) * NonLinearSystem.dalbedo(T)

    @staticmethod
    def Q_solar(x: np.ndarray) -> np.ndarray:
        return NonLinearSystem.Q_0 * (1 - 0.241 * (1 - x**2))

    @staticmethod
    def albedo(T: np.ndarray) -> np.ndarray:
        return NonLinearSystem.a_1 + (NonLinearSystem.a_2 - NonLinearSystem.a_1) / 2 * (
            1 + np.tanh(T - NonLinearSystem.T_star)
        )

    @staticmethod
    def dalbedo(T: np.ndarray) -> np.ndarray:
        return (NonLinearSystem.a_2 - NonLinearSystem.a_1) / (
            2 * np.cos(T - NonLinearSystem.T_star) ** 2
        )

    @staticmethod
    # Black body radiation (R_E)
    def R_E(T: np.ndarray) -> np.ndarray:
        return NonLinearSystem.epsilon_0 * NonLinearSystem.sigma_0 * T**4

    @staticmethod
    def dR_E(T: np.ndarray) -> np.ndarray:
        return 4 * NonLinearSystem.epsilon_0 * NonLinearSystem.sigma_0 * T**3

    # dispersion & legendre polynomials
    def R_D(self) -> np.ndarray:
        """
        Evaluate the dispersion term using the coefficients and eigenvalues
        of the respective Legendre polynomials.

        Returns:
        --------
        np.ndarray (m,): The dispersion term evaluated at the given coefficients.
            m is trhe number of sample points.
        """
        return (self.legendre_eigs * self.T_coeffs) @ self.legendre_polys

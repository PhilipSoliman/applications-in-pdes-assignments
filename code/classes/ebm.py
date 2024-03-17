import numpy as np
from scipy.special import legendre
from typing import Callable

# easy guassian quadrature integration
from scipy.integrate import fixed_quad as guassian_quadrature_integration

# gauss-legendre quadrature weights (more efficient as no for loops are used)
from numpy.polynomial.legendre import leggauss

from non_linear_system import NonLinearSystem

class EBM(NonLinearSystem):
    # problem constants and parameters (capitalized is default value)
    mu_default = 30.0  # greenhouse gas and particle parameter
    a_1 = 0.7  # albedo of water
    a_2 = 0.289  # albedo of ice
    T_star = 272  # freezing temperature
    delta = 0  # heat flux at poles
    Q_0 = 341.3  # solar radiation constant
    sigma_0 = 5.67e-8  # Stefan-Boltzmann constant
    epsilon_0 = 0.61  # emissivity of the Earth
    C_T = 5.0e8  # heat capacity of the Earth
    D_default = 0.3  # heat dispersion coefficient
    M = 0.1  # temperature scaling parameter

    def __init__(self, n_polys: int, n_quad_points: int, n_gridpoints: int) -> None:
        self.n_polys = n_polys
        self.n_quad_points = n_quad_points
        self.n_gridpoints = n_gridpoints
        self.legendre_polys_callable, self.legendre_eigs, self.legendre_norms = (
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
        self._T_coeffs[0] = (
            EBM.T_star
        )  # earth has a homogeneous temperature
        self._mu = EBM.mu_default
        self._D = EBM.D_default

    # legendre polynomials and eigenvalues
    def generate_legendre_polynomials(self) -> tuple[list[Callable], np.ndarray]:
        polys = [legendre(i) for i in range(self.n_polys)]
        eigs = np.array([-i * (i + 1) for i in range(self.n_polys)])
        norms = np.array([2 / (2 * i + 1) for i in range(self.n_polys)])
        return (polys, eigs, norms)

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
        T = self.T_eval()
        integrand = self.energy_balance(T)
        test_function = self.legendre_polys
        weak_form = integrand * test_function
        # return (weak_form @ self.quad_weights[:, None]).flatten()
        return weak_form.dot(self.quad_weights)

    def evaluate_derivative(self) -> np.ndarray:
        """

        Returns:
        --------
        np.ndarray: matrix such that out[i,j] = int_{-1}{1} [(lambda_j + dR_A(T) - dR_E(T)) * phi_j] * phi_i dx
        """
        T = self.T_eval()
        integrand = self.energy_balance_derivative(T)
        test_function = self.legendre_polys
        weak_form_derivative = np.einsum("js,is->ijs", integrand, test_function)
        # return np.einsum("ijs,s->ij", weak_form_derivative, self.quad_weights)
        return weak_form_derivative @ self.quad_weights

    def evaluate_derivative_finite_difference(self, h: float = 1e-6) -> np.ndarray:
        """
        Evaluate the derivative of the non-linear system of equations using finite difference.

        Returns:
        --------
        np.ndarray: matrix such that out[i,j] = (F(T(a_i + h))[j] - F(T)[j]) / h
        """
        F_T = self.evaluate()
        T_coeffs_plus_h = self.T_coeffs * np.ones((self.n_polys, self.n_polys)) + h * np.eye(self.n_polys)
        T_plus_h = T_coeffs_plus_h @ self.legendre_polys
        integrand = self.energy_balance(T_plus_h) # n x n_samples
        test_function = self.legendre_polys
        weak_form = np.einsum("is,js->ijs", integrand, test_function)
        F_T_plus_h = weak_form @ self.quad_weights
        return (F_T_plus_h - F_T[:, None]) / h
    
    def update_solution(self, update: np.ndarray) -> None:
        """
        Add the given update to current solution.
        """
        self.T_coeffs += update

    ######## Energy balance terms ########
    def energy_balance(self, T: np.ndarray) -> np.ndarray:
        return self.R_D() + self.R_A(self.quad_samples, T) - self.R_E(T)

    def energy_balance_derivative(self, T: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the energy balance terms w.r.t. to (the coefficients of) T
        using the coefficients and eigenvalues of the respective Legendre polynomials.

        The derivative of the dispersion term are the eigenvalues of the Laplacian
        multiplied by the Legendre polynomials (see dR_d method). The derivative of the solar radiation
        and the black body radiation are their derivative w.r.t. T multiplied by the
        Legendre polynomials (chain rule).

        Returns:
        --------
        np.ndarray (n, m): The derivative of the energy balance terms evaluated at the given coefficients.
            m is the number of sample points and n is the number of Legendre polynomials.
        """
        return (
            self.dR_D()
            + (self.dR_A(self.quad_samples, T) - self.dR_E(T)) * self.legendre_polys
        )

    # solar radiation (R_A)
    def R_A(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        # return EBM.Q_solar(x) * (1 - EBM.albedo(T)) + self.mu
        out = EBM.Q_solar(x) * (1 - EBM.albedo(T)) + self.mu
        # return np.zeros_like(out)
        return out

    @staticmethod
    def dR_A(x: np.ndarray, T: np.ndarray) -> np.ndarray:
        # return EBM.Q_solar(x) * EBM.dalbedo(T)
        out = EBM.Q_solar(x) * EBM.dalbedo(T)
        # return np.zeros_like(out)
        return out

    @staticmethod
    def Q_solar(x: np.ndarray) -> np.ndarray:
        return EBM.Q_0 * (1 - 0.241 * (3 * x**2 - 1))

    @staticmethod
    def albedo(T: np.ndarray) -> np.ndarray:
        return EBM.a_1 + (EBM.a_2 - EBM.a_1) / 2 * (
            1 + np.tanh(EBM.M * (T - EBM.T_star))
        )

    @staticmethod
    def dalbedo(T: np.ndarray) -> np.ndarray:
        return (
            EBM.M
            * (EBM.a_2 - EBM.a_1)
            / (2 * np.cos(EBM.M * (T - EBM.T_star)) ** 2)
        )

    @staticmethod
    # Black body radiation (R_E)
    def R_E(T: np.ndarray) -> np.ndarray:
        # return EBM.epsilon_0 * EBM.sigma_0 * T**4
        out = EBM.epsilon_0 * EBM.sigma_0 * T**4
        # return np.zeros_like(out)
        return out

    @staticmethod
    def dR_E(T: np.ndarray) -> np.ndarray:
        # return 4 * EBM.epsilon_0 * EBM.sigma_0 * T**3
        out = 4 * EBM.epsilon_0 * EBM.sigma_0 * T**3
        # return np.zeros_like(out)
        return out

    # dispersion & legendre polynomials
    def R_D(self) -> np.ndarray:
        """
        Evaluate the dispersion term using the coefficients and eigenvalues
        of the respective Legendre polynomials.

        Returns:
        --------
        np.ndarray (m,): The dispersion term evaluated for the current T_coeffs
            and at the m quad_sample points.
        """
        return (self.legendre_eigs / self.D * self.T_coeffs) @ self.legendre_polys

    def dR_D(self) -> np.ndarray:
        """
        Evaluate the derivative of the dispersion term using the coefficients and eigenvalues
        of the respective Legendre polynomials.

        Returns:
        --------
        np.ndarray (n, m): The derivative of the dispersion term evaluated at the given coefficients.
            m is the number of sample points and n is the number of Legendre polynomials.
        """
        return self.legendre_eigs[:, None] / self.D * self.legendre_polys

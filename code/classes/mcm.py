from typing import Callable

import numpy as np
import sympy as sym
from non_linear_system import NonLinearSystem

# gauss-legendre quadrature weights (more efficient as no for loops are used)
from numpy.polynomial.legendre import leggauss

# easy guassian quadrature integration
from scipy.integrate import fixed_quad as guassian_quadrature_integration
from scipy.special import legendre


class MCM(NonLinearSystem):

    # default parameters
    p1_default = 0.5
    p2_default = 2.5
    p3_default = 0.6
    p4_default = 1.5
    p5_default = 4.5
    p6_default = 1.0
    p7_default = 0.2
    p8_default = 0.5

    def __init__(self) -> None:
        self._p1 = self.p1_default
        self._p2 = self.p2_default
        self._p3 = self.p3_default
        self._p4 = self.p4_default
        self._p5 = self.p5_default
        self._p6 = self.p6_default
        self._p7 = self.p7_default
        self._p8 = self.p8_default

    # main methods
    def evaluate(self) -> np.ndarray:
        pass

    def evaluate_derivative(self) -> np.ndarray:
        pass

    def evaluate_derivative_finite_difference(self, h: float = 1e-6) -> np.ndarray:
        pass

    def get_current_solution(self) -> np.ndarray:
        pass

    def set_current_solution(self, solution: np.ndarray) -> None:
        pass

    def update_solution(self, update) -> None:
        pass

    # symbolic methods
    def getDimensionlessSystem(self) -> dict:
        t = sym.symbols("t")  # independent variable (time)
        T = sym.Function("T")(t)
        H = sym.Function("H")(t)
        E = sym.Function("E")(t)  # Tumor, Healthy & Effector cells
        k1, k2, k3 = sym.symbols(
            "k1 k2 k3"
        )  # coefficients: Tumor, Healthy & Effector cell carrying capacities
        r1, r2, r3 = sym.symbols("r1 r2 r3")  # growth rates
        a12, a13, a21, a31 = sym.symbols("a12 a13 a21 a31")  # interaction coefficients
        d3 = sym.symbols("d3")  # death rate of effector cells
        x1, x2, x3 = sym.symbols("x1 x2 x3")  # dimensionless variables
        p1, p2, p3, p4, p5, p6, p7, p8 = sym.symbols(
            "p1 p2 p3 p4 p5 p6 p7 p8"
        )  # dimensionless parameters
        T = k1 * x1
        H = k2 * x2
        E = k3 * x3
        rhs = [
            (f1 := r1 * T * (1 - T / k1) - a12 * T * H - a13 * T * E),
            (f2 := r2 * H * (1 - H / k2) - a21 * H * T),
            (f3 := r3 * T * E / (T + k3) - a31 * E * T - d3 * E),
        ]
        rhsDimensionless = [
            (f1_d := x1 * (1 - x1) - p1 * x1 * x2 - p2 * x1 * x3),
            (f2_d := p3 * x2 * (1 - x2) - p4 * x2 * x1),
            (f3_d := p5 * x1 * x3 / (x1 + p6) - p7 * x1 * x3 - p8 * x3),
        ]
        eqs = [sym.Eq(rhs[i], rhsDimensionless[i]) for i in range(len(rhs))]
        solution1 = sym.solve_undetermined_coeffs(
            eqs[0], [p1, p2], x1, x2, x3, dict=True
        )[0]
        solution2 = sym.solve_undetermined_coeffs(eqs[1], [p3, p4], x1, x2, dict=True)[
            0
        ]
        solution3 = sym.solve_undetermined_coeffs(
            eqs[2], [p5, p6, p7, p8], x1, x3, dict=True
        )[0]
        pvalues = {}
        params = [p1, p2, p3, p4, p5, p6, p7, p8]
        solutions = (
            list(solution1.values())
            + list(solution2.values())
            + list(solution3.values())
        )
        for parameter, solution in zip(params, solutions):
            pvalues[str(parameter)] = solution
        return {
            "dimensionless_system": rhsDimensionless,
            "parameters": pvalues,
        }

    # getters
    @property
    def p1(self) -> float:
        return self._p1

    @property
    def p2(self) -> float:
        return self._p2

    @property
    def p3(self) -> float:
        return self._p3

    @property
    def p4(self) -> float:
        return self._p4

    @property
    def p5(self) -> float:
        return self._p5

    @property
    def p6(self) -> float:
        return self._p6

    @property
    def p7(self) -> float:
        return self._p7

    @property
    def p8(self) -> float:
        return self._p8

    # setters
    @p1.setter
    def p1(self, value: float) -> None:
        self._p1 = value

    @p2.setter
    def p2(self, value: float) -> None:
        self._p2 = value

    @p3.setter
    def p3(self, value: float) -> None:
        self._p3 = value

    @p4.setter
    def p4(self, value: float) -> None:
        self._p4 = value

    @p5.setter
    def p5(self, value: float) -> None:
        self._p5 = value

    @p6.setter
    def p6(self, value: float) -> None:
        self._p6 = value

    @p7.setter
    def p7(self, value: float) -> None:
        self._p7 = value

    @p8.setter
    def p8(self, value: float) -> None:
        self._p8 = value

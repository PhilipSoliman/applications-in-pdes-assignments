from pprint import pprint
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

    # sympy print settings
    printSettings = dict(
        use_unicode=True,
        wrap_line=False,
        imaginary_unit="i",
        num_columns=70,
    )

    # constructor
    def __init__(self) -> None:
        self._p1 = self.p1_default
        self._p2 = self.p2_default
        self._p3 = self.p3_default
        self._p4 = self.p4_default
        self._p5 = self.p5_default
        self._p6 = self.p6_default
        self._p7 = self.p7_default
        self._p8 = self.p8_default
        self._x = None
        self.system = None
        self.systemDimensionless = None
        self.stationaryPoints = None
        self.defineSymbols()

    def defineSymbols(self) -> None:
        self.sym_t = sym.symbols("t")  # independent variable (time)
        self.sym_T = sym.Function("T")(self.sym_t)
        self.sym_H = sym.Function("H")(self.sym_t)
        self.sym_E = sym.Function("E")(self.sym_t)  # Tumor, Healthy & Effector cells
        self.sym_k1, self.sym_k2, self.sym_k3 = sym.symbols(
            "k1 k2 k3"
        )  # coefficients: Tumor, Healthy & Effector cell carrying capacities
        self.sym_r1, self.sym_r2, self.sym_r3 = sym.symbols("r1 r2 r3")  # growth rates
        self.sym_a12, self.sym_a13, self.sym_a21, self.sym_a31 = sym.symbols(
            "a12 a13 a21 a31"
        )  # interaction coefficients
        self.sym_d3 = sym.symbols("d3")  # death rate of effector cells
        self.sym_x1 = sym.Function("x1")(self.sym_t)
        self.sym_x2 = sym.Function("x2")(self.sym_t)
        self.sym_x3 = sym.Function("x3")(self.sym_t)  # dimensionless variables
        (
            self.sym_p1,
            self.sym_p2,
            self.sym_p3,
            self.sym_p4,
            self.sym_p5,
            self.sym_p6,
            self.sym_p7,
            self.sym_p8,
        ) = sym.symbols(
            "p1 p2 p3 p4 p5 p6 p7 p8"
        )  # dimensionless parameters
        self.sym_tau = sym.symbols(r"\tau")  # time scale

        # coordinate transformation
        # self.sym_tau = self.sym_r1 * self.sym_tau
        # self.sym_T = self.sym_k1 * self.sym_x1
        # self.sym_H = self.sym_k2 * self.sym_x2
        # self.sym_E = self.sym_k3 * self.sym_x3

    # main methods
    def evaluate(self) -> np.ndarray:
        return np.array(
            [
                self.x[0] * (1 - self.x[0])
                - self.p1 * self.x[0] * self.x[1]
                - self.p2 * self.x[0] * self.x[2],
                self.p3 * self.x[1] * (1 - self.x[1]) - self.p4 * self.x[1] * self.x[0],
                self.p5 * self.x[0] * self.x[2] / (self.x[0] + self.p6)
                - self.p7 * self.x[0] * self.x[2]
                - self.p8 * self.x[2],
            ]
        )

    def evaluate_derivative(self) -> np.ndarray:
        x1, x2, x3 = self.x
        return np.array(
            [
                [
                    -self.p1 * x2 - self.p2 * x3 - 2 * x1 + 1,
                    -self.p1 * x1,
                    -self.p2 * x1,
                ],
                [-self.p4 * x2, self.p3 * (1 - x2) - self.p3 * x2 - self.p4 * x1, 0],
                [
                    self.p5 * x3 / (self.p6 + x1)
                    - self.p5 * x1 * x3 / (self.p6 + x1) ** 2
                    - self.p7 * x3,
                    0,
                    self.p5 * x1 / (self.p6 + x1) - self.p7 * x1 - self.p8,
                ],
            ]
        )

    def evaluate_derivative_finite_difference(self, h: float = 1e-6) -> np.ndarray:
        pass

    def get_current_solution(self) -> np.ndarray:
        return self.x

    def set_current_solution(self, solution: np.ndarray) -> None:
        self.x = solution

    def update_solution(self, update) -> None:
        self.x += update

    # symbolic methods
    def constructSystem(self) -> None:
        rhs = [
            self.sym_r1 * self.sym_T * (1 - self.sym_T / self.sym_k1)
            - self.sym_a12 * self.sym_T * self.sym_H
            - self.sym_a13 * self.sym_T * self.sym_E,
            self.sym_r2 * self.sym_H * (1 - self.sym_H / self.sym_k2)
            - self.sym_a21 * self.sym_H * self.sym_T,
            self.sym_r3 * self.sym_T * self.sym_E / (self.sym_T + self.sym_k3)
            - self.sym_a31 * self.sym_E * self.sym_T
            - self.sym_d3 * self.sym_E,
        ]
        self.system = [
            sym.Eq(self.sym_T.diff(self.sym_t), rhs[0]),
            sym.Eq(self.sym_H.diff(self.sym_t), rhs[1]),
            sym.Eq(self.sym_E.diff(self.sym_t), rhs[2]),
        ]
        rhsDimensionless = [
            self.sym_x1 * (1 - self.sym_x1)
            - self.sym_p1 * self.sym_x1 * self.sym_x2
            - self.sym_p2 * self.sym_x1 * self.sym_x3,
            self.sym_p3 * self.sym_x2 * (1 - self.sym_x2)
            - self.sym_p4 * self.sym_x2 * self.sym_x1,
            self.sym_p5 * self.sym_x1 * self.sym_x3 / (self.sym_x1 + self.sym_p6)
            - self.sym_p7 * self.sym_x1 * self.sym_x3
            - self.sym_p8 * self.sym_x3,
        ]
        self.systemDimensionless = [
            sym.Eq(self.sym_x1.diff(self.sym_t), rhsDimensionless[0]),
            sym.Eq(self.sym_x2.diff(self.sym_t), rhsDimensionless[1]),
            sym.Eq(self.sym_x3.diff(self.sym_t), rhsDimensionless[2]),
        ]
        self.defineSystemJacobian()

    def defineSystemJacobian(self) -> None:
        self.J = sym.Matrix(
            [
                [eq.rhs.diff(self.sym_T) for eq in self.system],
                [eq.rhs.diff(self.sym_H) for eq in self.system],
                [eq.rhs.diff(self.sym_E) for eq in self.system],
            ]
        ).T
        self.J_dimensionless = sym.Matrix(
            [
                [eq.rhs.diff(self.sym_x1) for eq in self.systemDimensionless],
                [eq.rhs.diff(self.sym_x2) for eq in self.systemDimensionless],
                [eq.rhs.diff(self.sym_x3) for eq in self.systemDimensionless],
            ]
        ).T

    def printSystem(self) -> None:
        if self.system:
            print("Original System:")
            for eq in self.system:
                self.symPrint(eq)
            print("Jacobian:")
            self.symPrint(self.J)

    def printDimensionlessSystem(self) -> None:
        if self.systemDimensionless:
            print("Dimensionless System:")
            for eq in self.systemDimensionless:
                self.symPrint(eq)
            print("Jacobian:")
            self.symPrint(self.J_dimensionless)

    def findStationaryPoints(self) -> None:
        if self.systemDimensionless is None:
            self.constructSystem()
        pvalues = {
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
            "p4": self.p4,
            "p5": self.p5,
            "p6": self.p6,
            "p7": self.p7,
            "p8": self.p8,
        }
        sys = [eq.subs(pvalues) for eq in self.systemDimensionless]
        eqs = [sym.Eq(eq.rhs, 0) for eq in sys]
        stationary_points = sym.solve(eqs, (self.sym_x1, self.sym_x2, self.sym_x3))
        J = sym.Matrix(
            [
                [eq.rhs.diff(self.sym_x1) for eq in sys],
                [eq.rhs.diff(self.sym_x2) for eq in sys],
                [eq.rhs.diff(self.sym_x3) for eq in sys],
            ]
        )
        eigenvalues = []
        stability = []
        for x_0 in stationary_points:
            subs = {self.sym_x1: x_0[0], self.sym_x2: x_0[1], self.sym_x3: x_0[2]}
            J_eval = J.subs(subs)
            eigenvals = list(J_eval.eigenvals().keys())
            eigenvalues.append(eigenvals)
            if all([sym.re(eigenval) < 0 for eigenval in eigenvals]):
                stability.append("Stable")
            elif any([sym.re(eigenval) > 0 for eigenval in eigenvals]):
                stability.append("Unstable")
            else:
                stability.append("Saddle")

        self.stationaryPoints = dict(
            stationary_points=stationary_points,
            eigenvalues=eigenvalues,
            stability=stability,
        )

        # self.symPrint(self.stationaryPoints)

    def getStableStationaryPoints(self) -> list:
        if self.stationaryPoints is None:
            self.findStationaryPoints()
        stable_points = []
        for i, point in enumerate(self.stationaryPoints["stationary_points"]):
            if self.stationaryPoints["stability"][i] == "Stable":
                stable_points.append(np.array(list(point), dtype=float))
        return stable_points

    def symPrint(self, expr: Callable) -> None:
        sym.pprint(expr, **self.printSettings)

    # getters
    @property
    def x(self) -> np.ndarray:
        return self._x

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
    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._x = value

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

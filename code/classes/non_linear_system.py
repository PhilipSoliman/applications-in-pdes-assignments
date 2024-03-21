import numpy as np
from abc import ABC, abstractmethod


class NonLinearSystem(ABC):

    @abstractmethod
    def evaluate(self) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_derivative(self) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_derivative_finite_difference(self, h: float = 1e-6) -> np.ndarray:
        pass

    @abstractmethod
    def get_current_solution(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def update_solution(self, update) -> None:
        pass
from abc import ABC, abstractmethod

import numpy as np


class ModelABC(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        pass

from abc import ABC, abstractmethod
import numpy as np

class ModelABC(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        pass
    

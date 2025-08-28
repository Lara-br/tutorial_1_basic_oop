import numpy as np
from collections import Counter
from helper import *

class KNN:
    def __init__(self, k: int = 3):
        self.k = k
        self._parameters = {}
    
    def fit(self, observations: np.ndarray, labels: np.ndarray):
        self._parameters = {
            "observations" : observations,
            "labels" : labels
        }
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray) -> int:
        dists = np.linalg.norm(self._parameters["observations"] - x, axis=1)
        nn_indices = np.argsort(dists)[:self.k]
        
        nn_labels = self._parameters["labels"][nn_indices]
        most_common = Counter(nn_labels).most_common(1)
        return most_common[0][0]
    
    @property
    def observations(self) -> np.ndarray:
        return self._parameters["observations"].copy()
    
    @property
    def labels(self) -> np.ndarray:
        return self._parameters["labels"].copy()
    
data, labels, apples, oranges = generate_true_data(50)
new_fruit = generate_obs_data(10)

model = KNN()
model.fit(data, labels)

pred = model.predict(new_fruit)

plot_fruits(apples, oranges, new_fruit, pred)

plot_boundary(model, data)
import numpy as np
from baseclass import ModelABC
from helper import *


class KNN(ModelABC):
    def __init__(self, k: int = 3) -> None:
        pass

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        pass

    def predict(self, observations: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self) -> None:
        pass


data, labels, apples, oranges = generate_true_data(50)
new_fruit = generate_obs_data(10)

model = KNN()
model.fit(data, labels)

pred = model.predict(new_fruit)

plot_fruits(apples, oranges, new_fruit, pred)
plot_boundary(model, data)

import numpy as np
from baseclass import ModelABC
from helper import *


class KMeans(ModelABC):
    def __init__(self, k: int) -> None:
        self.k = k
        self.centroids = None

    def fit(
        self, observations: np.ndarray, threshold: float = 0.00001,
    ) -> list[np.ndarray]:
        pass

    def make_clusters(self, observations: np.ndarray) -> list[np.ndarray]:
        pass


data, labels, apples, oranges = generate_true_data(50)
cm = KMeans(2)
clusters = cm.fit(data)
plot_clusters(clusters)

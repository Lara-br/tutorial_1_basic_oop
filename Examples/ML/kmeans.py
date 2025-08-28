import numpy as np
from helper import *
import matplotlib.pyplot as plt
from baseclass import ModelABC


class KMeans(ModelABC):
    def __init__(self, k: int):
        self.k = k
        self.centroids = None
        
    def fit(self, observations: np.ndarray, threshold: float = 0.00001) -> np.ndarray:
        num_obs, dim = observations.shape
        centroids_idx = np.random.choice(num_obs, self.k)
        self.centroids = observations[centroids_idx]
        
        converged = False
        
        while not converged:
            clusters = self.make_clusters(observations)
            # self.plot_clusters(clusters)
            
            new_centroids = np.array([self._compute_centroid(c) for c in clusters])
            print(new_centroids, self.centroids, np.linalg.matrix_norm(self.centroids - new_centroids))
            
            if np.linalg.matrix_norm(self.centroids - new_centroids) < threshold:
                converged = True
            
            self.centroids = new_centroids
        return clusters
            
    def _compute_centroid(self, cluster: np.ndarray) -> np.ndarray:
        return np.mean(cluster, axis=0)
    
    def make_clusters(self, observations: np.ndarray) -> list[np.ndarray]:
        clusters = [[] for _ in range(self.k)]
        
        for x in observations:
            dists = np.linalg.norm(self.centroids - x, axis=1)
            closest_idx = np.argsort(dists)[0]
            clusters[closest_idx].append(x)
        return [np.array(c) for c in clusters]

        
    

data, labels, apples, oranges = generate_true_data(50)
cm = KMeans(2)
clusters = cm.fit(data)
plot_clusters(clusters)

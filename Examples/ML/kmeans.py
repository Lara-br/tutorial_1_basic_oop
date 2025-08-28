import numpy as np
from baseclass import ModelABC
from helper import *


class KMeans(ModelABC):
    def __init__(self, k: int) -> None:
        """
        Initializes the KMeans clustering instance.
        Args:
            k (int): The number of clusters to form.
        """

        self.k = k
        self.centroids = None

    def fit(self, observations: np.ndarray, threshold: float = 0.00001) -> np.ndarray:
        """
        Fits the KMeans model to the given observations using the Lloyd's algorithm.
        Args:
            observations (np.ndarray): A 2D numpy array of shape (num_samples, num_features) containing the data points to cluster.
            threshold (float, optional): The convergence threshold. The algorithm stops when the change in centroids is less than this value. Default is 0.00001.
        Returns:
            np.ndarray: A list or array of clusters, where each cluster contains the indices or data points assigned to it.
        """

        num_obs, dim = observations.shape
        centroids_idx = np.random.choice(num_obs, self.k)
        self.centroids = observations[centroids_idx]

        converged = False

        while not converged:
            clusters = self.make_clusters(observations)

            new_centroids = np.array([self._compute_centroid(c) for c in clusters])
            print(
                new_centroids,
                self.centroids,
                np.linalg.matrix_norm(self.centroids - new_centroids),
            )

            if np.linalg.matrix_norm(self.centroids - new_centroids) < threshold:
                converged = True

            self.centroids = new_centroids
        return clusters

    def _compute_centroid(self, cluster: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a given cluster.
        Args:
            cluster (np.ndarray): A 2D NumPy array where each row represents a data point in the cluster.
        Returns:
            np.ndarray: A 1D NumPy array representing the centroid (mean) of the cluster along each feature axis.
        """

        return np.mean(cluster, axis=0)

    def make_clusters(self, observations: np.ndarray) -> list[np.ndarray]:
        """
        Assigns each observation to the nearest centroid to form clusters.
        Args:
            observations (np.ndarray): A 2D array of shape (n_samples, n_features) containing the data points to be clustered.
        Returns:
            list[np.ndarray]: A list of length `k`, where each element is a numpy array containing the observations assigned to that cluster.
        """

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

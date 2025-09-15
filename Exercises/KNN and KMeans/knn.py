import numpy as np
from baseclass import ModelABC
from helper import *


class KNN(ModelABC):
    def __init__(self, k: int = 3) -> None:
        """
        Initializes the KNN classifier with the specified number of neighbors.
        Args:
            k (int, optional): The number of nearest neighbors to use for classification. Defaults to 3.
        """
        self.k: int = k
        self._parameters: dict = {}

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits the KNN model to the provided training data.
        Args:
            observations (np.ndarray): The feature matrix of training data, where each row represents an observation.
            labels (np.ndarray): The corresponding labels for each observation.
        Returns:
            None
        """
        self._parameters: dict = {"observations": observations, "labels": labels}

    def predict(self, observations: np.ndarray) -> np.ndarray:
         """
        Predicts the class labels for the given observations using the trained k-NN model.
        Args:
            observations (np.ndarray): A 2D numpy array where each row represents an observation to classify.
        Returns:
            np.ndarray: An array of predicted class labels for each observation.
        """
        predictions: list = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray) -> None:
        """
        Predicts the label for a single input sample using the k-nearest neighbors algorithm.
        Args:
            x (np.ndarray): The input sample for which to predict the label. Should be a 1D numpy array representing feature values.
        Returns:
            int: The predicted label for the input sample, determined by majority vote among the k nearest neighbors.
        """
        distance: int = np.linalg.normm(self._parameters["observations"] - x, axis=1)
        nn_indices = np.
        


data, labels, apples, oranges = generate_true_data(50)
new_fruit = generate_obs_data(10)

model = KNN()
model.fit(data, labels)

pred = model.predict(new_fruit)

plot_fruits(apples, oranges, new_fruit, pred)
plot_boundary(model, data)

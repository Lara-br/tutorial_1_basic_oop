import matplotlib.pyplot as plt
import numpy as np
from baseclass import ModelABC


def generate_true_data(samples_per_class) -> np.ndarray:
    """
    Generates a 2D NumPy array of random observation data.
    Each observation consists of two features, both uniformly distributed in the range [1, 9).
    Args:
        samples (int): The number of observations (rows) to generate.
    Returns:
        np
        .ndarray: A (samples, 2) array of generated observation data.
    """
    # Raw data
    apples = np.random.random((samples_per_class, 2))
    oranges = np.random.random((samples_per_class, 2))

    # Rescale variables
    apples[:, 0] = apples[:, 0] * 4 + 1  # range 1-5
    apples[:, 1] = apples[:, 1] * 3 + 6  # range 6-9

    oranges[:, 0] = oranges[:, 0] * 3 + 6  # range 6-9
    oranges[:, 1] = oranges[:, 1] * 4 + 1  # range 1-5

    data = np.concatenate((apples, oranges))
    labels = np.concatenate((np.zeros(samples_per_class), np.ones(samples_per_class)))
    return data, labels, apples, oranges


def generate_obs_data(samples) -> np.ndarray:
    """
    Generates a 2D NumPy array of random observation data.
    Each observation consists of two features, both uniformly distributed in the range [1, 9).
    Args:
        samples (int): The number of observations (rows) to generate.
    Returns:
        np.ndarray: A (samples, 2) array of generated observation data.
    """

    data = np.random.random((samples, 2))

    data[:, 0] = data[:, 0] * 8 + 1
    data[:, 1] = data[:, 1] * 8 + 1
    return data


def plot_fruits(apples: np.ndarray, oranges: np.ndarray, new_fruit: np.ndarray, prediction: np.ndarray) -> None:
    """
    Plots a scatter plot to visualize apples, oranges, and a new fruit with its predicted label.
    Args:
        apples (np.ndarray): 2D array of shape (n_apples, 2) representing apple features (size, color intensity).
        oranges (np.ndarray): 2D array of shape (n_oranges, 2) representing orange features (size, color intensity).
        new_fruit (np.ndarray): 2D array of shape (1, 2) representing the new fruit's features.
        prediction (list or np.ndarray): Predicted label for the new fruit (e.g., ["Apple"] or ["Orange"]).
    """

    # Generate the chart
    plt.figure(figsize=(10, 6))

    # Plot observed fruits
    plt.scatter(apples[:, 0], apples[:, 1], color="red", label="Apple", s=100)

    plt.scatter(oranges[:, 0], oranges[:, 1], color="orange", label="Orange", s=100)

    # Plot new fruit
    plt.scatter(
        new_fruit[0, 0],
        new_fruit[0, 1],
        color="green",
        label="New Fruit",
        s=200,
        marker="*",
    )

    plt.title("Fruit Classification: Apples vs Oranges")
    plt.xlabel("Size")
    plt.ylabel("Color Intensity")
    plt.legend()

    # Add text annotation for the prediction
    plt.annotate(
        f"Predicted: {prediction[0]}",
        xy=(new_fruit[0, 0], new_fruit[0, 1]),
        xytext=(5, 5.5),
        arrowprops={"facecolor": "black", "shrink": 0.05},
    )

    plt.grid(False)
    plt.show()


def plot_boundary(model: ModelABC, observations: np.ndarray) -> None:
    """
    Plots the decision boundary of a classification model for a 2D feature space.
    Args:
        model (ModelABC): A classification model implementing a `predict` method that accepts a 2D numpy array of observations.
        observations (np.ndarray): A 2D numpy array of shape (n_samples, 2) containing the feature values for plotting the boundary.
    """


    # Get bounds
    x1min, x1max = observations[:, 0].min() - 1, observations[:, 0].max() + 1
    x2min, x2max = observations[:, 1].min() - 1, observations[:, 1].max() + 1

    # Make grid
    x1_lin = np.linspace(x1min, x1max, 100)
    x2_lin = np.linspace(x2min, x2max, 100)
    x1, x2 = np.meshgrid(x1_lin, x2_lin)

    obs_grid = np.stack([x1.ravel(), x2.ravel()], axis=1)
    labels = model.predict(obs_grid)

    fig, ax = plt.subplots(figsize=(10, 6))

    contour = ax.contourf(x1, x2, -labels.reshape(x1.shape), alpha=0.7, cmap="autumn")

    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Apple", "Orange"])

    # Set axis labels and title
    ax.set_title("Decision Boundary: Apples vs Oranges", fontsize=16, fontweight="bold")
    ax.set_xlabel("Size", fontsize=12)
    ax.set_ylabel("Color Intensity", fontsize=12)

    plt.show()


def plot_clusters(self, clusters: np.ndarray) -> None:
    """
    Plots clusters of data points along with their centroids.
    Args:
        clusters (np.ndarray): A list or array of clusters, where each cluster is an array-like of points (shape: [n_points, 2]).
    """

    colors = ["r", "b", "g", "c", "m", "y", "k"]
    for idx, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        if cluster.size == 0:
            continue
        plt.scatter(
            cluster[:, 0],
            cluster[:, 1],
            color=colors[idx % len(colors)],
            label=f"Cluster {idx + 1}",
        )
    plt.scatter(
        self.centroids[:, 0],
        self.centroids[:, 1],
        color="k",
        marker="x",
        s=100,
        label="Centroids",
    )
    plt.legend()
    plt.show()

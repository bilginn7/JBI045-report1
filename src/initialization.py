import numpy as np
from src.lloyd import euclidean_distances_squared


def random_initialization(X, k, rng):
    """Select k random data points as initial centroids.

    :param X: Data matrix with shape (n_samples, n_features).
    :param k: Number of clusters.
    :param rng: Random number generator.
    :return: Initial centroids with shape (k, n_features).
    """
    if k > len(X):
        raise ValueError("k cannot be larger than the number of data points.")

    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def kmeans_plus_plus_initialization(X, k, rng):
    """Select initial centroids using the k-means++ method.

    The first centroid is selected uniformly at random. Each next centroid is
    selected with probability proportional to the squared distance from the
    nearest already chosen centroid.

    :param X: Data matrix with shape (n_samples, n_features).
    :param k: Number of clusters.
    :param rng: Random number generator.
    :return: Initial centroids with shape (k, n_features).
    """
    if k > len(X):
        raise ValueError("k cannot be larger than the number of data points.")

    n_samples = len(X)
    centers = []

    first_index = rng.integers(n_samples)
    centers.append(X[first_index])

    for _ in range(1, k):
        centers_array = np.array(centers)

        distances = euclidean_distances_squared(X, centers_array)
        nearest_distances = np.min(distances, axis=1)

        total_distance = np.sum(nearest_distances)

        if total_distance == 0:
            remaining_indices = rng.choice(n_samples, size=k - len(centers), replace=False)
            centers.extend(X[remaining_indices])
            break

        probabilities = nearest_distances / total_distance
        next_index = rng.choice(n_samples, p=probabilities)

        centers.append(X[next_index])

    return np.array(centers)


def gonzalez_initialization(X, k, rng):
    """Select initial centroids using Gonzalez farthest-first initialization.

    The first centroid is selected uniformly at random. Each next centroid is
    the point farthest away from its nearest already chosen centroid.

    :param X: Data matrix with shape (n_samples, n_features).
    :param k: Number of clusters.
    :param rng: Random number generator.
    :return: Initial centroids with shape (k, n_features).
    """
    if k > len(X):
        raise ValueError("k cannot be larger than the number of data points.")

    n_samples = len(X)
    centers = []

    first_index = rng.integers(n_samples)
    centers.append(X[first_index])

    for _ in range(1, k):
        centers_array = np.array(centers)

        distances = euclidean_distances_squared(X, centers_array)
        nearest_distances = np.min(distances, axis=1)

        next_index = np.argmax(nearest_distances)
        centers.append(X[next_index])

    return np.array(centers)
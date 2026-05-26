import numpy as np


def euclidean_distances_squared(X, centers):
    """Compute squared Euclidean distances from every point to every center.

    :param X: Data matrix with shape (n_samples, n_features).
    :param centers: Centroid matrix with shape (k, n_features).
    :return: Squared distance matrix with shape (n_samples, k).
    """
    return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def compute_inertia(X, centers, labels):
    """Compute the k-means clustering cost.

    Inertia is the sum of squared distances from each point to its assigned
    centroid.

    :param X: Data matrix with shape (n_samples, n_features).
    :param centers: Centroid matrix with shape (k, n_features).
    :param labels: Cluster assignment for each point.
    :return: Sum of squared distances to assigned centroids.
    """
    return float(np.sum((X - centers[labels]) ** 2))


def lloyd_kmeans(X, initial_centers, max_iter=300, tol=0.0):
    """Run Lloyd's k-means algorithm from fixed initial centroids.

    :param X: Data matrix with shape (n_samples, n_features).
    :param initial_centers: Initial centroids with shape (k, n_features).
    :param max_iter: Maximum number of Lloyd iterations.
    :param tol: Optional tolerance for centroid movement. If tol is 0.0,
        convergence is based only on unchanged assignments.
    :return: Dictionary containing final centers, labels, inertia, number of
        iterations, and inertia history.
    """
    centers = initial_centers.copy()
    old_labels = None
    inertia_history = []

    for iteration in range(1, max_iter + 1):
        distances = euclidean_distances_squared(X, centers)
        labels = np.argmin(distances, axis=1)

        inertia = compute_inertia(X, centers, labels)
        inertia_history.append(inertia)

        if old_labels is not None and np.array_equal(labels, old_labels):
            break

        new_centers = centers.copy()

        for j in range(len(centers)):
            cluster_points = X[labels == j]

            if len(cluster_points) > 0:
                new_centers[j] = cluster_points.mean(axis=0)

        center_shift = np.linalg.norm(new_centers - centers)

        centers = new_centers
        old_labels = labels.copy()

        if tol > 0.0 and center_shift <= tol:
            break

    distances = euclidean_distances_squared(X, centers)
    labels = np.argmin(distances, axis=1)
    final_inertia = compute_inertia(X, centers, labels)

    return {
        "centers": centers,
        "labels": labels,
        "inertia": final_inertia,
        "iterations": iteration,
        "inertia_history": inertia_history
    }
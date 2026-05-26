import numpy as np
from src.initialization import random_initialization, kmeans_plus_plus_initialization, gonzalez_initialization
from src.lloyd import lloyd_kmeans


def inertia_decreases(history):
    """Check if inertia does not increase."""
    for i in range(len(history) - 1):
        if history[i + 1] > history[i]:
            return False
    return True


if __name__ == "__main__":
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [5.0, 5.0],
        [5.2, 4.9],
        [10.0, 10.0],
        [10.1, 9.8],
    ])

    k = 3
    seed = 42

    methods = {
        "random": random_initialization,
        "k-means++": kmeans_plus_plus_initialization,
        "gonzalez": gonzalez_initialization,
    }

    for name, init_method in methods.items():
        rng = np.random.default_rng(seed)

        initial_centers = init_method(X, k, rng)
        result = lloyd_kmeans(X, initial_centers)

        print(f"\n{name}")
        print("centers shape:", result["centers"].shape)
        print("labels:", result["labels"])
        print("inertia:", result["inertia"])
        print("iterations:", result["iterations"])
        print("inertia decreases:", inertia_decreases(result["inertia_history"]))
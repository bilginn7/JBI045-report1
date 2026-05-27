import time
import numpy as np
import pandas as pd

from src.initialization import (
    random_initialization,
    kmeans_plus_plus_initialization,
    gonzalez_initialization,
)

from src.lloyd import lloyd_kmeans
from src.metrics import compute_silhouette


METHODS = {
    "random": random_initialization,
    "kmeans++": kmeans_plus_plus_initialization,
    "gonzalez": gonzalez_initialization,
}

def run_single_experiment(X, method_name, k, seed):

    rng = np.random.default_rng(seed)

    init_method = METHODS[method_name]

    start = time.perf_counter()

    initial_centers = init_method(X, k, rng)

    result = lloyd_kmeans(X, initial_centers)

    runtime = time.perf_counter() - start

    silhouette = compute_silhouette(X, result["labels"])

    return {
        "method": method_name,
        "k": k,
        "seed": seed,
        "inertia": result["inertia"],
        "iterations": result["iterations"],
        "silhouette": silhouette,
        "runtime": runtime,
    }

def run_experiments(X, k_values, n_runs):

    rows = []

    for method in METHODS:

        for k in k_values:

            for seed in range(n_runs):

                row = run_single_experiment(
                    X,
                    method,
                    k,
                    seed,
                )

                rows.append(row)

    return pd.DataFrame(rows)
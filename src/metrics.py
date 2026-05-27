import pandas as pd
from sklearn.metrics import silhouette_score


def compute_silhouette(X, labels):
    """
    Compute silhouette score.

    Returns NaN if only one cluster exists.
    """

    if len(set(labels)) < 2:
        return float("nan")

    return silhouette_score(X, labels)


def summarize_results(df):
    """
    Aggregate experiment results.
    """

    summary = (
        df.groupby(["method", "k"])
        .agg(
            inertia_mean=("inertia", "mean"),
            inertia_std=("inertia", "std"),
            silhouette_mean=("silhouette", "mean"),
            silhouette_std=("silhouette", "std"),
            iterations_mean=("iterations", "mean"),
            iterations_std=("iterations", "std"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
        )
        .reset_index()
    )

    return summary
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.config import K_VALUES, N_RUNS
from src.experiment import run_experiments
from src.metrics import summarize_results

from src.visualizations import (
    plot_inertia,
    plot_silhouette,
)

def load_dataset(path):

    df = pd.read_csv(path)

    X = df[
        [
            "annual_return",
            "annual_volatility",
        ]
    ].values

    scaler = StandardScaler()

    return scaler.fit_transform(X)

def main():

    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data"

    results_dir = root / "results"

    figs_dir = root / "figs"

    full_path = data_dir / "sp500_stats_2025.csv"

    clean_path = data_dir / "sp500_outliers_2025.csv"

    print("Loading datasets...")

    X_full = load_dataset(full_path)

    X_clean = load_dataset(clean_path)

    print("Running experiments (full dataset)...")

    full_results = run_experiments(
        X_full,
        K_VALUES,
        N_RUNS,
    )

    print("Running experiments (clean dataset)...")

    clean_results = run_experiments(
        X_clean,
        K_VALUES,
        N_RUNS,
    )

    full_summary = summarize_results(full_results)

    clean_summary = summarize_results(clean_results)

    (results_dir / "raw").mkdir(
        parents=True,
        exist_ok=True,
    )

    (results_dir / "summaries").mkdir(
        parents=True,
        exist_ok=True,
    )

    full_results.to_csv(
        results_dir / "raw" / "full_results.csv",
        index=False,
    )

    clean_results.to_csv(
        results_dir / "raw" / "clean_results.csv",
        index=False,
    )

    full_summary.to_csv(
        results_dir / "summaries" / "full_summary.csv",
        index=False,
    )

    clean_summary.to_csv(
        results_dir / "summaries" / "clean_summary.csv",
        index=False,
    )

    plot_inertia(
        full_summary,
        figs_dir / "inertia" / "full_inertia.png",
    )

    plot_silhouette(
        full_summary,
        figs_dir / "silhouette" / "full_silhouette.png",
    )

    plot_inertia(
        clean_summary,
        figs_dir / "inertia" / "clean_inertia.png",
    )

    plot_silhouette(
        clean_summary,
        figs_dir / "silhouette" / "clean_silhouette.png",
    )

    print("Finished.")


if __name__ == "__main__":
    main()
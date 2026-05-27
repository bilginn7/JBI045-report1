from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_inertia(summary, output_path):

    methods = summary["method"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:

        subset = summary[summary["method"] == method]

        ax.errorbar(
            subset["k"],
            subset["inertia_mean"],
            yerr=subset["inertia_std"],
            marker="o",
            label=method,
        )

    ax.set_title("Inertia vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()

def plot_silhouette(summary, output_path):

    methods = summary["method"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:

        subset = summary[summary["method"] == method]

        ax.errorbar(
            subset["k"],
            subset["silhouette_mean"],
            yerr=subset["silhouette_std"],
            marker="o",
            label=method,
        )

    ax.set_title("Silhouette Score vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()
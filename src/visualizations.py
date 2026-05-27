from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def plot_iterations(summary, output_path):

    methods = summary["method"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:

        subset = summary[summary["method"] == method]

        ax.errorbar(
            subset["k"],
            subset["iterations_mean"],
            yerr=subset["iterations_std"],
            marker="o",
            label=method,
        )

    ax.set_title("Average Lloyd Iterations vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Iterations")

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()

def plot_runtime(summary, output_path):

    methods = summary["method"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:

        subset = summary[summary["method"] == method]

        ax.errorbar(
            subset["k"],
            subset["runtime_mean"],
            yerr=subset["runtime_std"],
            marker="o",
            label=method,
        )

    ax.set_title("Average Runtime vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Runtime (seconds)")

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()

def plot_full_vs_clean(
    full_summary,
    clean_summary,
    metric,
    ylabel,
    output_path,
):

    methods = full_summary["method"].unique()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        sharey=True,
    )

    for ax, k in zip(axes, [3, 5, 8]):

        full_k = full_summary[
            full_summary["k"] == k
        ]

        clean_k = clean_summary[
            clean_summary["k"] == k
        ]

        x = np.arange(len(methods))
        width = 0.35

        full_vals = [
            full_k.loc[
                full_k["method"] == m,
                metric,
            ].iloc[0]
            for m in methods
        ]

        clean_vals = [
            clean_k.loc[
                clean_k["method"] == m,
                metric,
            ].iloc[0]
            for m in methods
        ]

        ax.bar(
            x - width / 2,
            full_vals,
            width,
            label="Full",
        )

        ax.bar(
            x + width / 2,
            clean_vals,
            width,
            label="Clean",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(methods)

        ax.set_title(f"k = {k}")

        ax.set_ylabel(ylabel)

    axes[0].legend()

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()

def plot_inertia_distribution(
    results,
    k,
    output_path,
):

    subset = results[
        results["k"] == k
    ]

    methods = subset["method"].unique()

    values = [
        subset[
            subset["method"] == m
        ]["inertia"]
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        values,
        labels=methods,
    )

    ax.set_title(
        f"Inertia Distribution (k={k})"
    )

    ax.set_ylabel("Inertia")

    fig.tight_layout()

    fig.savefig(output_path)

    plt.close()
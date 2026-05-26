from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


def setup_matplotlib() -> None:
    """Configure matplotlib to use LaTeX rendering with serif fonts."""
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot_return_vs_volatility(stats: pd.DataFrame, stats_out: pd.DataFrame, year: int, output_path: Path) -> None:
    """Plot annual return vs volatility, highlighting outliers.

    :param stats: Full stats DataFrame including outliers.
    :param stats_out: Stats DataFrame with outliers removed.
    :param year: Year of the data, used in title and output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(stats['annual_return'],
               stats['annual_volatility'],
               alpha=0.4,
               label='Outliers',
               color='red')
    ax.scatter(stats_out['annual_return'],
               stats_out['annual_volatility'],
               alpha=0.6,
               label='Clean',
               color='steelblue')

    ax.minorticks_on()
    ax.grid(True, which='major', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linewidth=0.2, alpha=0.4)

    ax.set_title(rf'S\&P 500 Annual Return vs Volatility ({year})', fontsize=16, fontweight='bold')
    ax.set_xlabel(r'Annual Return', fontsize=14)
    ax.set_ylabel(r'Annual Volatility', fontsize=14)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    setup_matplotlib()

    year: int = 2025

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figs_dir = project_root / "figs"

    stats_path = data_dir / f"sp500_stats_{year}.csv"
    stats_out_path = data_dir / f"sp500_outliers_{year}.csv"
    output_path = figs_dir / f"sp500_{year}.png"

    stats: pd.DataFrame = pd.read_csv(stats_path)
    stats_out: pd.DataFrame = pd.read_csv(stats_out_path)

    plot_return_vs_volatility(stats, stats_out, year, output_path)
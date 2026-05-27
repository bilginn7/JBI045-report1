# JBI045 – Report 1
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/bilginn7/JBI045-report1)

Implementation and experimental evaluation of Lloyd’s k-means clustering using different centroid initialization strategies on S&P 500 stock data.

Dataset source:
[Kaggle – S&P 500 Stocks Dataset](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily)

---

## Research Question
How do outlier stocks affect random, k-means++, and 
Gonzalez initialization when clustering S&P 500 stocks by return and volatility?

### Methods

We cluster stocks using Lloyd’s k-means with different initializations:
random, k-means++, and Gonzalez.

Each stock is represented using:

- Annual return
- Annualized volatility

The experiments compare clustering behavior on:

1. The original dataset containing outliers
2. A cleaned dataset where outliers are removed using the IQR method

The following metrics are evaluated:

- Inertia
- Silhouette score
- Number of Lloyd iterations
- Runtime

---

# Main Findings

- Gonzalez initialization is highly sensitive to outliers.
- Outliers artificially improve Gonzalez silhouette scores because extreme points become isolated clusters.
- k-means++ provides the most stable overall clustering behavior.
- Random initialization has the highest variance and convergence instability.
- After removing outliers, all methods perform much more similarly.

---

# Repository Structure

```text
├── src
│   ├── main.py                 # entry point / runs experiments
│   ├── experiment.py           # experiment runner
│   ├── initialization.py       # initialization methods
│   ├── lloyd.py                # Lloyd's algorithm implementation
│   ├── metrics.py              # evaluation metrics
│   ├── data_cleaning.py        # preprocessing and outlier removal
│   ├── figures.py              # figure generation helpers
│   ├── visualizations.py       # plotting functions
│   ├── config.py               # constants and experiment settings
│   └── __init__.py
├── data                        # input dataset
├── results                     # generated experiment results
├── figs                        # generated figures
├── test_algorithm.py           # simple algorithm test script
└── README.md
```

---

# Installation

## Requirements

- Python 3.11+
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

# Usage

Run the full experiment suite:

```bash
python src/main.py
```

Run the quick algorithm test:

```bash
python test_algorithm.py
```

Generated outputs include:

- CSV experiment results
- Cluster visualizations
- Evaluation figures

---

# Experimental Setup

The experiments were performed using:

- k values: 3, 5, and 8
- 30 repeated runs per method
- Standardized features
- Euclidean distance
- Lloyd’s iterative update procedure

Outliers were removed using the IQR rule:

:contentReference[oaicite:0]{index=0}

---

# Example Results

The report shows:

- Lower inertia after outlier removal
- Gonzalez producing higher silhouette scores on datasets containing extreme stocks
- Random initialization requiring the most Lloyd iterations
- k-means++ achieving the best stability-performance tradeoff

---

# Authors

- [Bilgin Eren](https://github.com/bilginn7)
- [Jan Austin Galić](https://github.com/jang1411)

---

# Authors

- [Bilgin Eren](https://www.github.com/bilginn7)
- [Jan Galic](https://www.github.com/jang1411)

---

# References

DarkMatterNet. *S&P 500 Stocks: 25 Years of Data (Updated Daily)*. Kaggle, 2024.

https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily

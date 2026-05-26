# JBI045 Report 1
![Gitea Last Commit](https://img.shields.io/gitea/last-commit/bilginn7/JBI045-report1)


Used the dataset from [Kaggle](https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily).


## Research Question
How do outlier stocks affect random, k-means++, and 
Gonzalez initialization when clustering S&P 500 stocks by return and volatility?

## Usage

```python
├── src
│   ├── initialization.py        # 3 initialization methods
│   └── lloyd.py                 # lloyd method algorithm
└── data
    ├── sp500_stats_2025.csv     # sp500 statistics dataset
    └── sp500_outliers_2025.csv  # dataset with outliers removed
```

## Authors

- [Bilgin Eren](https://www.github.com/bilginn7)
- [Jan Galic](https://www.github.com/jang1411)

from pathlib import Path
import pandas as pd


def clean_data(filepath: Path, year: int) -> pd.DataFrame:
    """Clean and filter the S&P 500 stocks dataset by specific year.

    :param filepath: Path to the S&P 500 stocks CSV file.
    :param year: Year to filter the data by.
    :return: Cleaned DataFrame with daily returns for the specified year.
    """
    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])

    # Filter specific year
    data = data.loc[(data['date'] >= f'{year}-01-01') & (data['date'] <= f'{year}-12-31')]

    # Compute pct_change per symbol
    data['daily_return'] = data.groupby('symbol')['close'].pct_change()

    return data


def compute_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Compute average daily return, volatility, and annualized stats per symbol.

    :param data: Cleaned DataFrame output from clean_data(), with a daily_return column.
    :return: DataFrame indexed by symbol with average return, volatility, n_days,
        annual return, and annual volatility.
    """
    stats = data.groupby('symbol')['daily_return'].agg(
        avg_daily_return='mean',
        volatility='std',
        n_days='count'
    )

    # Annualise using each symbol's own N
    stats['annual_return'] = stats['avg_daily_return'] * stats['n_days']
    stats['annual_volatility'] = stats['volatility'] * (stats['n_days'] ** 0.5)

    return stats

def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from the stats DataFrame using the IQR method.

    :param data: Stats DataFrame output from compute_stats().
    :return: Stats DataFrame with outliers removed.
    """
    result = data.copy()

    for col in ['annual_return', 'annual_volatility']:
        Q1 = result[col].quantile(0.25)
        Q3 = result[col].quantile(0.75)
        IQR = Q3 - Q1
        result = result.loc[(result[col] >= Q1 - 1.5 * IQR) & (result[col] <= Q3 + 1.5 * IQR)]

    return result

if __name__ == '__main__':
    year: int = 2025

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    input_path = data_dir / "sp500_stocks.csv"
    stats_path = data_dir / f"sp500_stats_{year}.csv"
    outliers_removed_path = data_dir / f"sp500_outliers_{year}.csv"

    cleaned_df: pd.DataFrame = clean_data(input_path, year)
    stats_df: pd.DataFrame = compute_stats(cleaned_df)
    outliers_removed_df: pd.DataFrame = remove_outliers(stats_df)

    print(stats_df)

    stats_df.to_csv(stats_path)
    outliers_removed_df.to_csv(outliers_removed_path)
import pandas as pd


def clean_data(filepath: str, year: int) -> pd.DataFrame:
    """Clean and filter the sp500 stocks dataset by specific year.

    :param filepath: Path to the sp500 stocks CSV file.
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
    """Compute average daily return, volatility, and annualised stats per symbol.
    Each symbol is annualised using its own number of trading days,
    making it suitable for stocks with partial year data.

    :param data: Cleaned DataFrame output from clean_data, with a daily_return column.
    :return: DataFrame indexed by symbol with the following columns:
        - avg_daily_return: Mean daily return.
        - volatility: Standard deviation of daily returns.
        - n_days: Number of trading days available for the symbol.
        - annual_return: Annualised return (avg_daily_return * n_days).
        - annual_volatility: Annualised volatility (volatility * sqrt(n_days)).
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


if __name__ == '__main__':

    filepath: str = 'data/sp500_stocks.csv'
    year: int = 2025

    cleaned_df: pd.DataFrame = clean_data(filepath, year)
    stats_df: pd.DataFrame = compute_stats(cleaned_df)

    print(stats_df)
    stats_df.to_csv(f'data/sp500_stats_{year}.csv')
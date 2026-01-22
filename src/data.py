"""Data fetching and preprocessing for portfolio optimization.

This module handles downloading historical stock data and calculating
the statistical metrics needed for Modern Portfolio Theory (MPT) analysis.
"""
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for given tickers.

    Uses Yahoo Finance to download historical price data. We use 'Adjusted Close'
    rather than raw 'Close' because it accounts for stock splits and dividends,
    giving a more accurate picture of total returns.

    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with dates as index and tickers as columns, containing
        adjusted close prices.

    Raises:
        ValueError: If no data is returned for the given tickers.
    """
    print("Downloading stock data...")
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    # Handle MultiIndex columns from yfinance (when downloading multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close']

    # Remove any rows with missing data
    data = data.dropna()

    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price data.

    Daily return is computed as the percentage change from the previous day:
        R_t = (P_t - P_{t-1}) / P_{t-1}

    This represents the simple return, which tells us the percentage gain
    or loss for each trading day.

    Args:
        prices: DataFrame of stock prices with dates as index.

    Returns:
        DataFrame of daily returns (same shape, minus first row which is NaN).
    """
    # pct_change() computes (current - previous) / previous
    returns = prices.pct_change().dropna()
    return returns


def calculate_metrics(returns: pd.DataFrame, trading_days: int = 252) -> tuple:
    """Calculate annualized return and risk metrics.

    Converts daily statistics to annual equivalents:
    - Mean return: multiply daily mean by trading days (assumes returns compound)
    - Covariance: multiply by trading days (variance scales linearly with time)
    - Correlation: no scaling needed (it's already normalized to [-1, 1])

    Args:
        returns: DataFrame of daily returns.
        trading_days: Number of trading days per year (default 252).

    Returns:
        Tuple of (mean_returns, cov_matrix, correlation):
        - mean_returns: Series of annualized expected returns per asset
        - cov_matrix: DataFrame of annualized covariance between assets
        - correlation: DataFrame of correlation coefficients between assets
    """
    # Annualize mean returns: daily_mean * 252
    mean_returns = returns.mean() * trading_days

    # Annualize covariance matrix: daily_cov * 252
    # Variance (and covariance) scales linearly with time for independent returns
    cov_matrix = returns.cov() * trading_days

    # Correlation is already normalized, no annualization needed
    correlation = returns.corr()

    return mean_returns, cov_matrix, correlation

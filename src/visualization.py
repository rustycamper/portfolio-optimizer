"""Visualization functions for portfolio analysis.

This module generates charts for analyzing portfolio optimization results,
including correlation matrices, price histories, and efficient frontiers.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_matrix(correlation: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save the correlation matrix heatmap.

    Correlation shows how assets move together (-1 to +1):
    - +1: Perfect positive correlation (move together)
    - 0: No correlation (independent movements)
    - -1: Perfect negative correlation (move opposite)

    Lower correlations between assets provide better diversification benefits.

    Args:
        correlation: Correlation matrix DataFrame.
        output_dir: Directory to save the output image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Stock Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300)
    plt.close()


def plot_price_history(prices: pd.DataFrame, output_dir: Path) -> None:
    """Plot normalized price history for all assets.

    Normalizes all prices to start at 100 for easy comparison
    of relative performance over time.

    Args:
        prices: DataFrame of stock prices with dates as index.
        output_dir: Directory to save the output image.
    """
    # Normalize to base 100 for comparison
    data_normalized = prices.div(prices.iloc[0]) * 100

    plt.figure(figsize=(12, 6))
    data_normalized.plot(ax=plt.gca())
    plt.title('Normalized Price History (Base 100)')
    plt.ylabel('Normalized Price')
    plt.xlabel('Date')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "price_history.png", dpi=300)
    plt.close()


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    optimal_point: tuple[float, float],
    equal_weight_point: tuple[float, float],
    output_dir: Path
) -> None:
    """Plot the efficient frontier with key portfolios marked.

    The efficient frontier shows the best possible risk-return tradeoffs.
    Points below the line are suboptimal; points above are unattainable.

    Args:
        frontier_df: DataFrame with 'return' and 'volatility' columns.
        individual_df: DataFrame with individual asset stats.
        optimal_point: (volatility, return) for max Sharpe portfolio.
        equal_weight_point: (volatility, return) for equal-weight portfolio.
        output_dir: Directory to save the output image.
    """
    plt.figure(figsize=(12, 8))

    # Plot the efficient frontier curve
    plt.plot(
        frontier_df['volatility'] * 100,
        frontier_df['return'] * 100,
        'b-', linewidth=2, label='Efficient Frontier'
    )

    # Plot individual assets
    plt.scatter(
        individual_df['volatility'] * 100,
        individual_df['return'] * 100,
        marker='o', s=200, c='red', edgecolors='black', linewidth=2,
        label='Individual Assets', zorder=3
    )

    # Label each individual asset
    for _, row in individual_df.iterrows():
        plt.annotate(
            row['ticker'],
            xy=(row['volatility'] * 100, row['return'] * 100),
            xytext=(10, 5), textcoords='offset points',
            fontsize=10, fontweight='bold'
        )

    # Plot optimal portfolio (max Sharpe)
    opt_vol, opt_ret = optimal_point
    plt.scatter(
        opt_vol * 100, opt_ret * 100,
        marker='*', s=500, c='gold', edgecolors='black', linewidth=2,
        label='Optimal Portfolio (Max Sharpe)', zorder=4
    )

    # Plot equal-weight portfolio
    eq_vol, eq_ret = equal_weight_point
    plt.scatter(
        eq_vol * 100, eq_ret * 100,
        marker='D', s=200, c='green', edgecolors='black', linewidth=2,
        label='Equal-Weight Portfolio', zorder=4
    )

    plt.xlabel('Annual Volatility (Risk) %', fontsize=12)
    plt.ylabel('Annual Return %', fontsize=12)
    plt.title('Efficient Frontier - Portfolio Optimization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "efficient_frontier.png", dpi=300)
    plt.close()


def plot_allocation_pie(
    weights: np.ndarray,
    tickers: list[str],
    output_dir: Path
) -> None:
    """Plot portfolio allocation as a pie chart.

    Only shows assets with weight > 1% to avoid clutter.

    Args:
        weights: Array of portfolio weights.
        tickers: List of ticker symbols.
        output_dir: Directory to save the output image.
    """
    # Filter out near-zero weights for cleaner visualization
    non_zero_weights = [
        (ticker, weight)
        for ticker, weight in zip(tickers, weights)
        if weight > 0.01
    ]

    if not non_zero_weights:
        return

    labels, weight_values = zip(*non_zero_weights)
    colors = plt.cm.Set3(range(len(labels)))

    plt.figure(figsize=(10, 8))
    plt.pie(weight_values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_allocation.png", dpi=300)
    plt.close()


def plot_risk_return_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot side-by-side bar charts comparing returns and volatility.

    Args:
        comparison_df: DataFrame with 'Asset', 'Return', 'Volatility' columns.
        output_dir: Directory to save the output image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme: blue for individual assets, gold for optimal, green for equal-weight
    num_assets = len(comparison_df) - 2  # Subtract optimal and equal-weight
    colors_bar = ['skyblue'] * num_assets + ['gold', 'green']

    # Return comparison
    ax1.bar(comparison_df['Asset'], comparison_df['Return'], color=colors_bar, edgecolor='black')
    ax1.set_ylabel('Annual Return %', fontsize=11)
    ax1.set_title('Expected Returns Comparison', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Volatility comparison
    ax2.bar(comparison_df['Asset'], comparison_df['Volatility'], color=colors_bar, edgecolor='black')
    ax2.set_ylabel('Annual Volatility %', fontsize=11)
    ax2.set_title('Risk (Volatility) Comparison', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "risk_return_comparison.png", dpi=300)
    plt.close()


def plot_sharpe_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Sharpe ratio comparison across all assets and portfolios.

    Sharpe ratio measures risk-adjusted return. A ratio > 1 is generally
    considered good; > 2 is very good; > 3 is excellent.

    Args:
        comparison_df: DataFrame with 'Asset' and 'Sharpe' columns.
        output_dir: Directory to save the output image.
    """
    num_assets = len(comparison_df) - 2
    colors_bar = ['skyblue'] * num_assets + ['gold', 'green']

    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df['Asset'], comparison_df['Sharpe'], color=colors_bar, edgecolor='black')
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.title('Sharpe Ratio Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1.0')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sharpe_comparison.png", dpi=300)
    plt.close()

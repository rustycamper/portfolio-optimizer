"""Portfolio optimizer entry point.

This script orchestrates the portfolio optimization workflow:
1. Fetch historical stock data
2. Calculate risk/return metrics
3. Find optimal portfolio weights (max Sharpe ratio)
4. Generate the efficient frontier
5. Create visualizations
6. Save results
"""
import numpy as np
import pandas as pd

from config import (
    TICKERS, RISK_FREE_RATE, MAX_WEIGHT, MIN_WEIGHT,
    USE_MAX_CONSTRAINT, USE_MIN_CONSTRAINT,
    START_DATE, END_DATE, OUTPUT_DIR, TRADING_DAYS_PER_YEAR
)
from data import fetch_stock_data, calculate_returns, calculate_metrics
from optimization import (
    portfolio_stats, optimize_max_sharpe, generate_efficient_frontier
)
from visualization import (
    plot_correlation_matrix, plot_price_history,
    plot_efficient_frontier, plot_allocation_pie,
    plot_risk_return_comparison, plot_sharpe_comparison
)


def main():
    """Run the complete portfolio optimization workflow."""

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ========== PHASE 1: GET DATA ==========
    prices = fetch_stock_data(TICKERS, START_DATE, END_DATE)
    returns = calculate_returns(prices)

    # ========== PHASE 2: CALCULATE METRICS ==========
    mean_returns, cov_matrix, correlation = calculate_metrics(
        returns, TRADING_DAYS_PER_YEAR
    )

    print("\n" + "=" * 50)
    print("INDIVIDUAL ASSET STATISTICS")
    print("=" * 50)

    volatility_annual = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_individual = (mean_returns - RISK_FREE_RATE) / volatility_annual

    stats_df = pd.DataFrame({
        'Annual Return': mean_returns * 100,
        'Annual Volatility': volatility_annual * 100,
        'Sharpe Ratio': sharpe_individual
    })
    print(stats_df)

    # ========== PHASE 3: PORTFOLIO OPTIMIZATION ==========

    # Determine constraints based on configuration
    max_w = MAX_WEIGHT if USE_MAX_CONSTRAINT else None
    min_w = MIN_WEIGHT if USE_MIN_CONSTRAINT else None

    # Run optimization
    optimal_result = optimize_max_sharpe(
        mean_returns, cov_matrix, RISK_FREE_RATE,
        max_weight=max_w, min_weight=min_w
    )

    if not optimal_result.success:
        print("\n⚠ WARNING: Optimization did not converge!")
        print(f"Message: {optimal_result.message}")

    optimal_weights = optimal_result.x
    opt_return, opt_volatility, opt_sharpe = portfolio_stats(
        optimal_weights, mean_returns, cov_matrix, RISK_FREE_RATE
    )

    ticker_names = prices.columns.tolist()

    print("\n" + "=" * 50)
    print("OPTIMAL PORTFOLIO")
    print("=" * 50)
    print("\nOptimal Weights:")
    for ticker, weight in zip(ticker_names, optimal_weights):
        if weight > 0.001:
            print(f"{ticker:6s}: {weight*100:6.2f}%")
        else:
            print(f"{ticker:6s}: {weight*100:6.2f}% (effectively zero)")

    print(f"\nExpected Annual Return: {opt_return*100:.2f}%")
    print(f"Annual Volatility (Risk): {opt_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {opt_sharpe:.3f}")

    # Calculate equal-weight portfolio for comparison
    equal_weights = np.array([1 / len(TICKERS)] * len(TICKERS))
    eq_return, eq_volatility, eq_sharpe = portfolio_stats(
        equal_weights, mean_returns, cov_matrix, RISK_FREE_RATE
    )

    print("\n" + "=" * 50)
    print("EQUAL-WEIGHT PORTFOLIO (for comparison)")
    print("=" * 50)
    print(f"Expected Annual Return: {eq_return*100:.2f}%")
    print(f"Annual Volatility (Risk): {eq_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {eq_sharpe:.3f}")

    # ========== PHASE 4: GENERATE EFFICIENT FRONTIER ==========
    efficient_df = generate_efficient_frontier(
        mean_returns, cov_matrix, RISK_FREE_RATE,
        max_weight=max_w, min_weight=min_w, n_points=50
    )

    # Individual asset stats for plotting
    individual_stats = []
    for i, ticker in enumerate(ticker_names):
        weights = np.zeros(len(ticker_names))
        weights[i] = 1.0
        ret, vol, sharpe = portfolio_stats(
            weights, mean_returns, cov_matrix, RISK_FREE_RATE
        )
        individual_stats.append({
            'ticker': ticker,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        })

    individual_df = pd.DataFrame(individual_stats)

    # ========== PHASE 5: CREATE VISUALIZATIONS ==========
    plot_correlation_matrix(correlation, OUTPUT_DIR)
    plot_price_history(prices, OUTPUT_DIR)
    plot_efficient_frontier(
        efficient_df, individual_df,
        optimal_point=(opt_volatility, opt_return),
        equal_weight_point=(eq_volatility, eq_return),
        output_dir=OUTPUT_DIR
    )
    plot_allocation_pie(optimal_weights, ticker_names, OUTPUT_DIR)

    # Comparison data for bar charts
    comparison_data = pd.DataFrame({
        'Asset': list(ticker_names) + ['Optimal', 'Equal-Weight'],
        'Return': list(individual_df['return'] * 100) + [opt_return * 100, eq_return * 100],
        'Volatility': list(individual_df['volatility'] * 100) + [opt_volatility * 100, eq_volatility * 100],
        'Sharpe': list(individual_df['sharpe']) + [opt_sharpe, eq_sharpe]
    })

    plot_risk_return_comparison(comparison_data, OUTPUT_DIR)
    plot_sharpe_comparison(comparison_data, OUTPUT_DIR)

    # ========== PHASE 6: SAVE RESULTS ==========
    optimal_portfolio_df = pd.DataFrame({
        'Ticker': ticker_names,
        'Weight': optimal_weights,
        'Weight_Percent': optimal_weights * 100
    })
    optimal_portfolio_df.to_csv(OUTPUT_DIR / "optimal_portfolio_weights.csv", index=False)

    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)
    print(f"\n✓ All files saved to {OUTPUT_DIR}/ folder")


if __name__ == "__main__":
    main()

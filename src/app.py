"""Streamlit web app for portfolio optimization.

This module provides an interactive web interface for the portfolio optimizer,
allowing users to select tickers, adjust parameters, and view results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from config import (
    TICKERS, RISK_FREE_RATE, MAX_WEIGHT, MIN_WEIGHT,
    START_DATE, END_DATE, TRADING_DAYS_PER_YEAR
)
from data import fetch_stock_data, calculate_returns, calculate_metrics
from optimization import (
    portfolio_stats, optimize_max_sharpe, generate_efficient_frontier
)
from visualization_plotly import (
    plot_correlation_heatmap, plot_price_history,
    plot_efficient_frontier, plot_allocation_pie,
    plot_risk_return_bars, plot_sharpe_comparison
)


st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)


@st.cache_data(show_spinner=False, ttl=3600)
def validate_ticker(ticker: str) -> bool:
    """Check if a ticker symbol is valid by querying Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        # Check if we got valid data (invalid tickers return minimal info)
        return info.get('regularMarketPrice') is not None or info.get('previousClose') is not None
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def get_stock_data(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch stock data with caching to avoid re-downloading."""
    return fetch_stock_data(list(tickers), start, end)


def main():
    st.title("Portfolio Optimizer")
    st.markdown("*Modern Portfolio Theory (MPT) optimization with interactive charts*")

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("Settings")

        # Ticker selection
        st.subheader("Ticker Selection")

        # Initialize session state for custom tickers
        if 'custom_tickers' not in st.session_state:
            st.session_state.custom_tickers = []

        # Preset tickers + any validated custom ones
        preset_tickers = TICKERS + ['AMZN', 'META', 'TSLA', 'BRK-B', 'V', 'MA', 'HD', 'UNH']
        available_tickers = sorted(set(preset_tickers + st.session_state.custom_tickers))

        selected_tickers = st.multiselect(
            "Select stocks",
            options=available_tickers,
            default=TICKERS,
            help="Select at least 2 stocks for portfolio optimization"
        )

        # Add custom ticker input
        if 'ticker_input' not in st.session_state:
            st.session_state.ticker_input = ""

        def add_ticker():
            ticker = st.session_state.ticker_input.upper().strip()
            if not ticker:
                return
            if ticker in available_tickers:
                st.toast(f":orange[**{ticker}** is already in the list]", icon=":material/info:")
            elif validate_ticker(ticker):
                st.session_state.custom_tickers.append(ticker)
                st.toast(f":green[**{ticker}** added to available stocks]", icon=":material/check_circle:")
            else:
                st.toast(f":red[**{ticker}** is not a valid ticker symbol]", icon=":material/error:")
            st.session_state.ticker_input = ""

        st.text_input(
            "Add custom ticker",
            placeholder="e.g., NFLX",
            help="Enter a ticker symbol and press Enter to validate",
            key="ticker_input",
            on_change=add_ticker
        )

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=pd.to_datetime(START_DATE),
                max_value=pd.to_datetime("today")
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=pd.to_datetime(END_DATE),
                max_value=pd.to_datetime("today")
            )

        # Constraints
        st.subheader("Constraints")
        use_max_weight = st.checkbox("Use maximum weight constraint", value=True)
        max_weight_pct = st.slider(
            "Max weight per asset",
            min_value=10,
            max_value=100,
            value=int(MAX_WEIGHT * 100),
            step=5,
            disabled=not use_max_weight,
            format="%d%%",
            help="Maximum allocation to any single asset"
        )
        max_weight = max_weight_pct / 100

        use_min_weight = st.checkbox("Use minimum weight constraint", value=False)
        min_weight_pct = st.slider(
            "Min weight per asset",
            min_value=0,
            max_value=20,
            value=int(MIN_WEIGHT * 100),
            step=1,
            disabled=not use_min_weight,
            format="%d%%",
            help="Minimum allocation to each asset (forces diversification)"
        )
        min_weight = min_weight_pct / 100

        # Risk-free rate
        risk_free_rate = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=RISK_FREE_RATE * 100,
            step=0.1,
            help="Annual risk-free rate (e.g., Treasury bill rate)"
        ) / 100

        # Run button
        st.divider()
        run_optimization = st.button(
            "Run Optimization",
            type="primary",
            use_container_width=True,
            disabled=len(selected_tickers) < 2
        )

        if len(selected_tickers) < 2:
            st.warning("Select at least 2 stocks")

    # ========== MAIN AREA ==========
    if not run_optimization and 'results' not in st.session_state:
        st.info("Configure settings in the sidebar and click **Run Optimization** to start.")
        return

    if run_optimization:
        # Validate constraints
        if use_min_weight and min_weight * len(selected_tickers) > 1:
            st.error(
                f"Infeasible constraints: {len(selected_tickers)} assets x "
                f"{min_weight*100:.0f}% min = {min_weight*len(selected_tickers)*100:.0f}% > 100%"
            )
            return

        with st.spinner("Fetching stock data..."):
            try:
                prices = get_stock_data(
                    tuple(sorted(selected_tickers)),
                    str(start_date),
                    str(end_date)
                )
            except ValueError as e:
                st.error(f"Error fetching data: {e}")
                return

        with st.spinner("Calculating metrics..."):
            returns = calculate_returns(prices)
            mean_returns, cov_matrix, correlation = calculate_metrics(
                returns, TRADING_DAYS_PER_YEAR
            )

        with st.spinner("Running optimization..."):
            max_w = max_weight if use_max_weight else None
            min_w = min_weight if use_min_weight else None

            try:
                optimal_result = optimize_max_sharpe(
                    mean_returns, cov_matrix, risk_free_rate,
                    max_weight=max_w, min_weight=min_w
                )
            except ValueError as e:
                st.error(f"Optimization failed: {e}")
                return

            if not optimal_result.success:
                st.warning(f"Optimization may not have converged: {optimal_result.message}")

            optimal_weights = optimal_result.x
            opt_return, opt_volatility, opt_sharpe = portfolio_stats(
                optimal_weights, mean_returns, cov_matrix, risk_free_rate
            )

            # Equal-weight portfolio
            equal_weights = np.array([1 / len(selected_tickers)] * len(selected_tickers))
            eq_return, eq_volatility, eq_sharpe = portfolio_stats(
                equal_weights, mean_returns, cov_matrix, risk_free_rate
            )

        with st.spinner("Generating efficient frontier..."):
            efficient_df = generate_efficient_frontier(
                mean_returns, cov_matrix, risk_free_rate,
                max_weight=max_w, min_weight=min_w, n_points=50
            )

        if efficient_df.empty:
            st.warning("Could not generate efficient frontier with current constraints. Try relaxing the weight constraints.")

        # Individual asset stats
        ticker_names = prices.columns.tolist()
        individual_stats = []
        volatility_annual = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        for i, ticker in enumerate(ticker_names):
            weights = np.zeros(len(ticker_names))
            weights[i] = 1.0
            ret, vol, sharpe = portfolio_stats(
                weights, mean_returns, cov_matrix, risk_free_rate
            )
            individual_stats.append({
                'ticker': ticker,
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            })

        individual_df = pd.DataFrame(individual_stats)

        # Comparison data
        comparison_df = pd.DataFrame({
            'Asset': list(ticker_names) + ['Optimal', 'Equal-Weight'],
            'Return': list(individual_df['return'] * 100) + [opt_return * 100, eq_return * 100],
            'Volatility': list(individual_df['volatility'] * 100) + [opt_volatility * 100, eq_volatility * 100],
            'Sharpe': list(individual_df['sharpe']) + [opt_sharpe, eq_sharpe]
        })

        # Store results in session state
        st.session_state['results'] = {
            'prices': prices,
            'correlation': correlation,
            'efficient_df': efficient_df,
            'individual_df': individual_df,
            'optimal_weights': optimal_weights,
            'opt_return': opt_return,
            'opt_volatility': opt_volatility,
            'opt_sharpe': opt_sharpe,
            'eq_return': eq_return,
            'eq_volatility': eq_volatility,
            'eq_sharpe': eq_sharpe,
            'comparison_df': comparison_df,
            'ticker_names': ticker_names,
            'risk_free_rate': risk_free_rate
        }

    # Display results from session state
    results = st.session_state['results']

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Price History", "Optimization", "Comparison"])

    with tab1:
        st.subheader("Price History")
        st.plotly_chart(
            plot_price_history(results['prices']),
            use_container_width=True
        )

        st.subheader("Correlation Matrix")
        st.plotly_chart(
            plot_correlation_heatmap(results['correlation']),
            use_container_width=True
        )

    with tab2:
        # Portfolio metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Expected Return",
                f"{results['opt_return']*100:.2f}%",
                delta=f"{(results['opt_return'] - results['eq_return'])*100:.2f}% vs equal-weight"
            )
        with col2:
            st.metric(
                "Volatility (Risk)",
                f"{results['opt_volatility']*100:.2f}%",
                delta=f"{(results['opt_volatility'] - results['eq_volatility'])*100:.2f}% vs equal-weight",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['opt_sharpe']:.3f}",
                delta=f"{results['opt_sharpe'] - results['eq_sharpe']:.3f} vs equal-weight"
            )

        st.subheader("Efficient Frontier")
        st.plotly_chart(
            plot_efficient_frontier(
                results['efficient_df'],
                results['individual_df'],
                optimal=(results['opt_volatility'], results['opt_return'], results['optimal_weights']),
                equal_weight=(results['eq_volatility'], results['eq_return']),
                tickers=results['ticker_names']
            ),
            use_container_width=True
        )

        st.subheader("Optimal Portfolio Allocation")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                plot_allocation_pie(results['optimal_weights'], results['ticker_names']),
                use_container_width=True
            )
        with col2:
            st.markdown("**Weights:**")
            weights_df = pd.DataFrame({
                'Ticker': results['ticker_names'],
                'Weight': [f"{w*100:.2f}%" for w in results['optimal_weights']]
            })
            weights_df = weights_df[results['optimal_weights'] > 0.001]
            st.dataframe(weights_df, hide_index=True, use_container_width=True)

    with tab3:
        st.subheader("Risk vs Return")
        st.plotly_chart(
            plot_risk_return_bars(results['comparison_df']),
            use_container_width=True
        )

        st.subheader("Sharpe Ratio Comparison")
        st.plotly_chart(
            plot_sharpe_comparison(results['comparison_df']),
            use_container_width=True
        )

    # Download section
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Download Results")
    with col2:
        csv_data = pd.DataFrame({
            'Ticker': results['ticker_names'],
            'Weight': results['optimal_weights'],
            'Weight_Percent': results['optimal_weights'] * 100
        }).to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="optimal_portfolio_weights.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

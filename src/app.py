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

# Preset portfolio configurations
PRESET_PORTFOLIOS = {
    "Custom": [],
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Dividend Aristocrats": ["JNJ", "PG", "KO", "PEP", "MCD", "WMT"],
    "FAANG+": ["META", "AAPL", "AMZN", "NFLX", "GOOGL", "MSFT"],
    "Blue Chips": ["AAPL", "MSFT", "JNJ", "JPM", "V", "PG", "UNH", "HD"],
    "Diversified Mix": ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "KO", "DIS"],
}

# Benchmark tickers
BENCHMARKS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
}


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


@st.cache_data(show_spinner=False)
def get_benchmark_stats(start: str, end: str, risk_free_rate: float) -> dict:
    """Fetch and calculate benchmark statistics."""
    benchmarks = {}
    for name, ticker in BENCHMARKS.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if data.empty:
                continue
            prices = data['Adj Close']
            returns = prices.pct_change().dropna()
            annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
            annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpe = (annual_return - risk_free_rate) / annual_vol
            benchmarks[name] = {
                'return': annual_return,
                'volatility': annual_vol,
                'sharpe': sharpe
            }
        except Exception:
            continue
    return benchmarks


def main():
    st.title("Portfolio Optimizer — Learn to Invest Smarter")
    st.caption("An educational app built by Lilly and [Claude Code](https://claude.ai/code)")
    st.markdown(
        "Discover how to build a smart investment portfolio using **Modern Portfolio Theory** (MPT) — "
        "a Nobel Prize-winning approach to balancing risk and reward."
    )
    with st.expander("What is this app? (click to learn more)"):
        st.markdown("""
        **Welcome!** This app helps you understand how professional investors build portfolios.

        **Key concepts you'll explore:**
        - **Diversification**: Don't put all your eggs in one basket — spreading investments reduces risk
        - **Risk vs Return**: Higher potential returns usually come with higher risk
        - **Sharpe Ratio**: A score that measures how much return you get for the risk you take (higher = better)
        - **Efficient Frontier**: The "sweet spot" portfolios that give you the best return for each level of risk

        **How to use:**
        1. Pick some stocks in the sidebar (or use a preset portfolio)
        2. Click "Run Optimization"
        3. Explore the interactive charts!

        📚 [Read the full documentation (PDF)](https://github.com/rustycamper/portfolio-optimizer/blob/main/Portfolio_Optimization_Documentation.pdf) for the math behind the scenes.

        *Built by Lilly and Claude Code as an educational project.*
        """)

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("Settings")

        # Ticker selection
        st.subheader("Ticker Selection")

        # Initialize session state for custom tickers and selected tickers
        if 'custom_tickers' not in st.session_state:
            st.session_state.custom_tickers = []
        if 'selected_tickers' not in st.session_state:
            st.session_state.selected_tickers = TICKERS.copy()
        if 'last_preset' not in st.session_state:
            st.session_state.last_preset = "Custom"

        # Preset tickers + any validated custom ones
        base_tickers = TICKERS + ['AMZN', 'META', 'TSLA', 'BRK-B', 'V', 'MA', 'HD', 'UNH', 'NFLX', 'PEP', 'MCD', 'WMT']
        available_tickers = sorted(set(base_tickers + st.session_state.custom_tickers))

        # Preset portfolio dropdown
        selected_preset = st.selectbox(
            "Preset portfolios",
            options=list(PRESET_PORTFOLIOS.keys()),
            index=0,
            help="Quick-start with a predefined portfolio, or choose 'Custom' to build your own"
        )

        # Update selection when preset changes
        if selected_preset != st.session_state.last_preset:
            st.session_state.last_preset = selected_preset
            if selected_preset != "Custom":
                st.session_state.selected_tickers = PRESET_PORTFOLIOS[selected_preset].copy()

        # Add custom ticker input first (before multiselect) so callback can update state
        if 'ticker_input' not in st.session_state:
            st.session_state.ticker_input = ""
        if 'pending_ticker' not in st.session_state:
            st.session_state.pending_ticker = None

        def add_ticker():
            ticker = st.session_state.ticker_input.upper().strip()
            if not ticker:
                return
            if ticker in st.session_state.selected_tickers:
                st.toast(f":orange[**{ticker}** is already selected]", icon=":material/info:")
            elif ticker in available_tickers:
                # Already in dropdown - add to selection
                st.session_state.pending_ticker = ticker
                st.toast(f":green[**{ticker}** added to selection]", icon=":material/check_circle:")
            elif validate_ticker(ticker):
                st.session_state.custom_tickers.append(ticker)
                st.session_state.pending_ticker = ticker
                st.toast(f":green[**{ticker}** added to selection]", icon=":material/check_circle:")
            else:
                st.toast(f":red[**{ticker}** is not a valid ticker symbol]", icon=":material/error:")
            st.session_state.ticker_input = ""

        # Process pending ticker addition
        if st.session_state.pending_ticker:
            if st.session_state.pending_ticker not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(st.session_state.pending_ticker)
            st.session_state.pending_ticker = None

        # Recalculate available tickers (may have changed if custom ticker was added)
        available_tickers = sorted(set(base_tickers + st.session_state.custom_tickers))

        selected_tickers = st.multiselect(
            "Select stocks",
            options=available_tickers,
            default=st.session_state.selected_tickers,
            help="Pick at least 2 stocks. More stocks = more diversification!"
        )

        # Sync selection back to session state
        st.session_state.selected_tickers = selected_tickers

        st.text_input(
            "Add custom ticker",
            placeholder="e.g., NFLX",
            help="Enter a ticker symbol and press Enter to validate and add to selection",
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
            help="Prevents putting too much money in one stock. Lower = more diversified."
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
            help="Forces every stock to have at least this much. Ensures no stock is ignored."
        )
        min_weight = min_weight_pct / 100

        # Risk-free rate
        risk_free_rate = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=RISK_FREE_RATE * 100,
            step=0.1,
            help="The return on a 'risk-free' investment like US Treasury bills. Used to calculate the Sharpe ratio."
        ) / 100

        # Benchmark toggle
        st.subheader("Benchmarks")
        show_benchmarks = st.checkbox(
            "Compare with benchmarks",
            value=True,
            help="Compare your portfolio against S&P 500 (SPY) and Nasdaq 100 (QQQ)"
        )

        # Run button
        st.divider()

        # Validation
        has_errors = False
        if len(selected_tickers) < 2:
            st.warning("Pick at least 2 stocks to build a portfolio!")
            has_errors = True
        if start_date >= end_date:
            st.error("Oops! Start date needs to be before end date.")
            has_errors = True

        run_optimization = st.button(
            "Run Optimization",
            type="primary",
            use_container_width=True,
            disabled=has_errors
        )

    # ========== MAIN AREA ==========
    if not run_optimization and 'results' not in st.session_state:
        st.info("👈 Pick your stocks in the sidebar, then click **Run Optimization** to see the magic happen!")
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

        # Fetch benchmark data if enabled
        benchmark_stats = {}
        if show_benchmarks:
            with st.spinner("Fetching benchmark data..."):
                benchmark_stats = get_benchmark_stats(
                    str(start_date), str(end_date), risk_free_rate
                )

        # Comparison data
        assets = list(ticker_names) + ['Optimal', 'Equal-Weight']
        returns_list = list(individual_df['return'] * 100) + [opt_return * 100, eq_return * 100]
        vol_list = list(individual_df['volatility'] * 100) + [opt_volatility * 100, eq_volatility * 100]
        sharpe_list = list(individual_df['sharpe']) + [opt_sharpe, eq_sharpe]

        # Add benchmarks to comparison
        for name, stats in benchmark_stats.items():
            assets.append(name)
            returns_list.append(stats['return'] * 100)
            vol_list.append(stats['volatility'] * 100)
            sharpe_list.append(stats['sharpe'])

        comparison_df = pd.DataFrame({
            'Asset': assets,
            'Return': returns_list,
            'Volatility': vol_list,
            'Sharpe': sharpe_list
        })

        # Store results in session state (including settings used)
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
            'risk_free_rate': risk_free_rate,
            'benchmark_stats': benchmark_stats,
            # Track settings used for this run
            'settings': {
                'tickers': set(selected_tickers),
                'start_date': str(start_date),
                'end_date': str(end_date),
            }
        }

    # Display results from session state
    results = st.session_state['results']

    # Check if settings have changed since last run
    if 'settings' in results:
        current_settings = {
            'tickers': set(selected_tickers),
            'start_date': str(start_date),
            'end_date': str(end_date),
        }
        if current_settings != results['settings']:
            st.warning("Settings have changed. Click **Run Optimization** to update results.", icon=":material/sync:")

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
                delta=f"{(results['opt_return'] - results['eq_return'])*100:.2f}% vs equal-weight",
                help="How much your portfolio might grow in a year (based on past performance)"
            )
        with col2:
            st.metric(
                "Volatility (Risk)",
                f"{results['opt_volatility']*100:.2f}%",
                delta=f"{(results['opt_volatility'] - results['eq_volatility'])*100:.2f}% vs equal-weight",
                delta_color="inverse",
                help="How bumpy the ride is — higher volatility means bigger ups and downs"
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['opt_sharpe']:.3f}",
                delta=f"{results['opt_sharpe'] - results['eq_sharpe']:.3f} vs equal-weight",
                help="Return per unit of risk — like miles per gallon for investments. Above 1 is good, above 2 is great!"
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
            # Create weights dataframe
            weights_data = []
            for ticker, weight in zip(results['ticker_names'], results['optimal_weights']):
                if weight > 0.001:
                    weights_data.append({
                        'Ticker': ticker,
                        'Weight': f"{weight*100:.2f}%"
                    })
            weights_df = pd.DataFrame(weights_data)
            st.dataframe(
                weights_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                    'Weight': st.column_config.TextColumn('Weight', width='small')
                }
            )

            # Download button
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
                icon=":material/download:",
                use_container_width=True
            )

    with tab3:
        # Color legend
        st.caption(
            ":blue-background[Individual Stocks]  "
            ":orange-background[Optimal Portfolio]  "
            ":green-background[Equal-Weight]  "
            ":violet-background[Benchmarks]"
        )

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


if __name__ == "__main__":
    main()

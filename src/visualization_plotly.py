"""Interactive Plotly visualizations for portfolio analysis.

This module provides interactive charts for the Streamlit web app,
replacing matplotlib with Plotly for better interactivity.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap.

    Args:
        correlation: Correlation matrix DataFrame.

    Returns:
        Plotly Figure object.
    """
    fig = px.imshow(
        correlation,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect='auto'
    )
    fig.update_layout(
        title='Stock Correlation Matrix',
        xaxis_title='',
        yaxis_title='',
        height=450
    )
    return fig


def plot_price_history(prices: pd.DataFrame) -> go.Figure:
    """Create an interactive price history chart.

    Normalizes all prices to start at 100 for easy comparison.

    Args:
        prices: DataFrame of stock prices with dates as index.

    Returns:
        Plotly Figure object.
    """
    # Normalize to base 100 for comparison
    normalized = prices.div(prices.iloc[0]) * 100

    fig = go.Figure()
    for column in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[column],
            mode='lines',
            name=column,
            hovertemplate='%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
        ))

    fig.update_layout(
        title='Normalized Price History (Base 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        hovermode='x unified',
        height=450,
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    optimal: tuple[float, float, np.ndarray],
    equal_weight: tuple[float, float],
    tickers: list[str]
) -> go.Figure:
    """Create an interactive efficient frontier chart.

    Args:
        frontier_df: DataFrame with 'return', 'volatility', 'weights' columns.
        individual_df: DataFrame with individual asset stats.
        optimal: Tuple of (volatility, return, weights) for optimal portfolio.
        equal_weight: Tuple of (volatility, return) for equal-weight portfolio.
        tickers: List of ticker symbols.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Efficient frontier line with hover showing weights (if data available)
    if not frontier_df.empty:
        hover_texts = []
        for _, row in frontier_df.iterrows():
            weights_text = '<br>'.join([
                f'{ticker}: {w*100:.1f}%'
                for ticker, w in zip(tickers, row['weights'])
                if w > 0.01
            ])
            hover_texts.append(
                f"Return: {row['return']*100:.2f}%<br>"
                f"Volatility: {row['volatility']*100:.2f}%<br>"
                f"Sharpe: {row['sharpe']:.3f}<br><br>"
                f"<b>Weights:</b><br>{weights_text}"
            )

        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'] * 100,
            y=frontier_df['return'] * 100,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    # Individual assets
    fig.add_trace(go.Scatter(
        x=individual_df['volatility'] * 100,
        y=individual_df['return'] * 100,
        mode='markers+text',
        name='Individual Assets',
        marker=dict(size=12, color='red', line=dict(width=2, color='black')),
        text=individual_df['ticker'],
        textposition='top right',
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Return: %{y:.2f}%<br>'
            'Volatility: %{x:.2f}%<br>'
            '<extra></extra>'
        )
    ))

    # Optimal portfolio
    opt_vol, opt_ret, opt_weights = optimal
    opt_weights_text = '<br>'.join([
        f'{ticker}: {w*100:.1f}%'
        for ticker, w in zip(tickers, opt_weights)
        if w > 0.01
    ])
    fig.add_trace(go.Scatter(
        x=[opt_vol * 100],
        y=[opt_ret * 100],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='black')),
        hovertext=f'<b>Optimal Portfolio</b><br>Return: {opt_ret*100:.2f}%<br>'
                  f'Volatility: {opt_vol*100:.2f}%<br><br><b>Weights:</b><br>{opt_weights_text}',
        hoverinfo='text'
    ))

    # Equal-weight portfolio
    eq_vol, eq_ret = equal_weight
    fig.add_trace(go.Scatter(
        x=[eq_vol * 100],
        y=[eq_ret * 100],
        mode='markers',
        name='Equal-Weight Portfolio',
        marker=dict(size=15, color='green', symbol='diamond', line=dict(width=2, color='black')),
        hovertemplate=(
            '<b>Equal-Weight Portfolio</b><br>'
            'Return: %{y:.2f}%<br>'
            'Volatility: %{x:.2f}%<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title='Efficient Frontier - Portfolio Optimization',
        xaxis_title='Annual Volatility (Risk) %',
        yaxis_title='Annual Return %',
        height=450,
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        hovermode='closest'
    )
    return fig


def plot_allocation_pie(weights: np.ndarray, tickers: list[str]) -> go.Figure:
    """Create an interactive pie chart of portfolio allocation.

    Args:
        weights: Array of portfolio weights.
        tickers: List of ticker symbols.

    Returns:
        Plotly Figure object.
    """
    # Filter out near-zero weights
    data = [(ticker, weight) for ticker, weight in zip(tickers, weights) if weight > 0.01]

    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No significant allocations", showarrow=False)
        return fig

    labels, values = zip(*data)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.2%}<extra></extra>',
        hole=0.3
    )])

    fig.update_layout(
        title='Optimal Portfolio Allocation',
        height=450
    )
    return fig


def _get_bar_colors(comparison_df: pd.DataFrame) -> list[str]:
    """Generate colors for bar charts based on asset type."""
    colors = []
    for asset in comparison_df['Asset']:
        if asset == 'Optimal':
            colors.append('#FFD700')  # Gold
        elif asset == 'Equal-Weight':
            colors.append('#32CD32')  # Green
        elif asset in ('S&P 500', 'Nasdaq 100'):
            colors.append('#9370DB')  # Purple for benchmarks
        else:
            colors.append('#87CEEB')  # Light blue for stocks
    return colors


def plot_risk_return_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar charts comparing returns and volatility.

    Args:
        comparison_df: DataFrame with 'Asset', 'Return', 'Volatility' columns.

    Returns:
        Plotly Figure object.
    """
    colors = _get_bar_colors(comparison_df)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Return (%)',
        x=comparison_df['Asset'],
        y=comparison_df['Return'],
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Volatility (%)',
        x=comparison_df['Asset'],
        y=comparison_df['Volatility'],
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Volatility: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Risk vs Return Comparison',
        xaxis_title='Asset',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=450,
        legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_sharpe_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart comparing Sharpe ratios.

    Args:
        comparison_df: DataFrame with 'Asset' and 'Sharpe' columns.

    Returns:
        Plotly Figure object.
    """
    colors = _get_bar_colors(comparison_df)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df['Asset'],
        y=comparison_df['Sharpe'],
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=1,
        hovertemplate='<b>%{x}</b><br>Sharpe Ratio: %{y:.3f}<extra></extra>'
    ))

    # Add reference line at Sharpe = 1.0
    fig.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='green',
        annotation_text='Sharpe = 1.0',
        annotation_position='right'
    )

    fig.update_layout(
        title='Sharpe Ratio Comparison (Higher is Better)',
        xaxis_title='Asset',
        yaxis_title='Sharpe Ratio',
        height=450
    )
    fig.update_xaxes(tickangle=-45)
    return fig

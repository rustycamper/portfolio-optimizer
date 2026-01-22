"""Portfolio optimization using Modern Portfolio Theory (MPT).

This module implements the core mathematical optimization for finding
optimal portfolio weights. Key concepts:

- **Portfolio Return**: R_p = Σ(w_i × r_i) - weighted sum of individual returns
- **Portfolio Variance**: σ²_p = w^T × Σ × w - quadratic form with covariance matrix
- **Sharpe Ratio**: (R_p - R_f) / σ_p - excess return per unit of risk
- **Efficient Frontier**: set of portfolios with minimum variance for each return level
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_stats(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float
) -> tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe ratio.

    Args:
        weights: Array of portfolio weights (must sum to 1).
        mean_returns: Expected annual returns for each asset.
        cov_matrix: Annualized covariance matrix of returns.
        risk_free_rate: Annual risk-free rate (e.g., Treasury bill rate).

    Returns:
        Tuple of (portfolio_return, portfolio_volatility, sharpe_ratio).
    """
    # Portfolio return: R_p = Σ(w_i × r_i)
    # The expected return is simply the weighted average of individual returns
    portfolio_return = np.dot(weights, mean_returns)

    # Portfolio variance: σ²_p = w^T × Σ × w
    # This captures how assets move together (covariance), not just individual risk
    # Diversification benefit comes from assets with low/negative correlation
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Sharpe ratio: (R_p - R_f) / σ_p
    # Measures excess return (above risk-free rate) per unit of risk
    # Higher is better: more return for the same amount of risk
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float
) -> float:
    """Return negative Sharpe ratio for minimization.

    Scipy's minimize function finds the minimum, but we want to maximize
    the Sharpe ratio. By negating it, minimizing negative Sharpe = maximizing Sharpe.

    Args:
        weights: Array of portfolio weights.
        mean_returns: Expected annual returns for each asset.
        cov_matrix: Annualized covariance matrix.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Negative Sharpe ratio (to be minimized).
    """
    return -portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def optimize_max_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float | None = None,
    min_weight: float | None = None
):
    """Find the portfolio with maximum Sharpe ratio.

    Uses Sequential Least Squares Programming (SLSQP) to find optimal weights
    subject to constraints (weights sum to 1, optional min/max bounds).

    Args:
        mean_returns: Expected annual returns for each asset.
        cov_matrix: Annualized covariance matrix.
        risk_free_rate: Annual risk-free rate.
        max_weight: Optional maximum weight per asset (e.g., 0.3 for 30%).
        min_weight: Optional minimum weight per asset (e.g., 0.05 for 5%).

    Returns:
        scipy.optimize.OptimizeResult with optimal weights in result.x

    Raises:
        ValueError: If min_weight constraints are mathematically infeasible.
    """
    num_assets = len(mean_returns)

    # Start with equal weights as initial guess
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Validate constraint feasibility
    # If min_weight * num_assets > 1, it's impossible to satisfy
    if min_weight is not None and min_weight * num_assets > 1:
        raise ValueError(
            f"Infeasible: {num_assets} assets × {min_weight*100}% min = "
            f"{min_weight*num_assets*100}% > 100%"
        )

    # Constraint: weights must sum to 1 (fully invested portfolio)
    # Using default arg to avoid Python closure late-binding bug
    constraints = [{'type': 'eq', 'fun': lambda w, mr=mean_returns: np.sum(w) - 1}]

    # Set bounds based on constraints
    if min_weight is not None and max_weight is not None:
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        print(f"\nOptimizing with constraints: {min_weight*100:.0f}% ≤ weight ≤ {max_weight*100:.0f}%")
    elif max_weight is not None:
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        print(f"\nOptimizing with constraint: weight ≤ {max_weight*100:.0f}%")
    elif min_weight is not None:
        bounds = tuple((min_weight, 1) for _ in range(num_assets))
        print(f"\nOptimizing with constraint: weight ≥ {min_weight*100:.0f}%")
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))
        print("\nOptimizing without weight constraints (allows concentration)")

    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    return result


def optimize_for_target_return(
    target_return: float,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float | None = None,
    min_weight: float | None = None
):
    """Find minimum variance portfolio for a given target return.

    This is used to generate the efficient frontier: for each possible
    return level, find the portfolio with the lowest risk (variance).

    Args:
        target_return: The desired annual portfolio return.
        mean_returns: Expected annual returns for each asset.
        cov_matrix: Annualized covariance matrix.
        max_weight: Optional maximum weight per asset.
        min_weight: Optional minimum weight per asset.

    Returns:
        scipy.optimize.OptimizeResult with optimal weights in result.x
    """
    num_assets = len(mean_returns)
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Two constraints:
    # 1. Weights sum to 1 (fully invested)
    # 2. Portfolio return equals target (using default args to avoid closure bug)
    constraints = [
        {'type': 'eq', 'fun': lambda w, mr=mean_returns: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, mr=mean_returns, tr=target_return: np.dot(w, mr) - tr}
    ]

    # Set bounds based on constraints
    if min_weight is not None and max_weight is not None:
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    elif max_weight is not None:
        bounds = tuple((0, max_weight) for _ in range(num_assets))
    elif min_weight is not None:
        bounds = tuple((min_weight, 1) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))

    # Objective: minimize portfolio variance (w^T × Σ × w)
    result = minimize(
        lambda w, cm=cov_matrix: np.dot(w, np.dot(cm, w)),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9}
    )

    return result


def generate_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float | None = None,
    min_weight: float | None = None,
    n_points: int = 50
) -> pd.DataFrame:
    """Generate the efficient frontier.

    The efficient frontier is the set of optimal portfolios that offer
    the highest expected return for a given level of risk (or equivalently,
    the lowest risk for a given return).

    Portfolios below the frontier are suboptimal (you could get more return
    for the same risk). Portfolios above the frontier are unattainable.

    Args:
        mean_returns: Expected annual returns for each asset.
        cov_matrix: Annualized covariance matrix.
        risk_free_rate: Annual risk-free rate.
        max_weight: Optional maximum weight per asset.
        min_weight: Optional minimum weight per asset.
        n_points: Number of points to generate on the frontier.

    Returns:
        DataFrame with columns: return, volatility, sharpe, weights
    """
    print("\nGenerating efficient frontier...")

    # Range of target returns: from minimum to maximum individual asset return
    min_return = mean_returns.min()
    max_return = mean_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    efficient_portfolios = []

    for target in target_returns:
        try:
            result = optimize_for_target_return(
                target, mean_returns, cov_matrix,
                max_weight=max_weight, min_weight=min_weight
            )
            if result.success:
                weights = result.x
                ret, vol, sharpe = portfolio_stats(
                    weights, mean_returns, cov_matrix, risk_free_rate
                )
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
        except Exception:
            # Some target returns may be infeasible given constraints
            continue

    efficient_df = pd.DataFrame(efficient_portfolios)
    print(f"✓ Generated {len(efficient_df)} efficient portfolios")

    return efficient_df

"""Portfolio optimizer configuration.

This module contains all configurable settings for the portfolio optimization
including asset selection, risk parameters, and output paths.
"""
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Constants
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year

# Portfolio settings
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG', 'JPM', 'XOM', 'DIS', 'NVDA', 'KO']
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate (e.g., Treasury bills)

# Weight constraints for diversification
MAX_WEIGHT = 0.30  # No more than 30% in any single asset
MIN_WEIGHT = 0.05  # At least 5% in each asset (forces diversification)
USE_MAX_CONSTRAINT = True   # Set to True to enforce max weight
USE_MIN_CONSTRAINT = False  # Set to True to enforce minimum weights

# Data settings
START_DATE = '2020-01-01'
END_DATE = '2026-01-01'

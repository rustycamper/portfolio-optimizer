# Portfolio Optimizer

Portfolio optimization tool implementing Modern Portfolio Theory (MPT).

For a detailed explanation of the math and methodology, see [Portfolio_Optimization_Documentation.pdf](Portfolio_Optimization_Documentation.pdf).

## Features
- Downloads historical stock data via Yahoo Finance
- Calculates risk/return metrics
- Finds optimal portfolio weights (max Sharpe ratio)
- Generates efficient frontier
- Creates visualizations

## Project Structure

```
src/
├── config.py        # Settings and constants (tickers, risk parameters)
├── data.py          # Fetch and preprocess stock data
├── optimization.py  # Portfolio optimization math (MPT algorithms)
├── visualization.py # Chart generation (efficient frontier, allocations)
├── main.py          # Entry point - orchestrates the workflow
└── output/          # Generated files (charts, CSV)
```

## Setup

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
brew install uv  # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS

# Run the optimizer
uv sync
uv run python src/main.py
```

## Configuration
Edit settings in `src/config.py`:
- `TICKERS`: List of stock symbols
- `RISK_FREE_RATE`: Risk-free rate (default: 0.02)
- `MAX_WEIGHT`: Maximum weight per asset (default: 0.30)
- `MIN_WEIGHT`: Minimum weight per asset (default: 0.05)
- `USE_MAX_CONSTRAINT`: Enable max weight constraint (default: True)
- `USE_MIN_CONSTRAINT`: Enable min weight constraint (default: False)
- `START_DATE` / `END_DATE`: Date range for historical data

## Output
Results are saved to `src/output/`:
- `optimal_portfolio_weights.csv`
- `efficient_frontier.png`
- `correlation_matrix.png`
- `price_history.png`
- `portfolio_allocation.png`
- `risk_return_comparison.png`
- `sharpe_comparison.png`

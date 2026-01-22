# Portfolio Optimizer

Portfolio optimization tool implementing Modern Portfolio Theory (MPT).

For a detailed explanation of the math and methodology, see [Portfolio_Optimization_Documentation.pdf](Portfolio_Optimization_Documentation.pdf).

## Live Demo

Try the interactive web app: **[Portfolio Optimizer on Streamlit](https://your-app-name.streamlit.app)** *(update link after deployment)*

## Features
- Downloads historical stock data via Yahoo Finance
- Calculates risk/return metrics
- Finds optimal portfolio weights (max Sharpe ratio)
- Generates efficient frontier
- Creates visualizations
- Interactive web app with Plotly charts
- CLI for batch processing

## Project Structure

```
src/
├── app.py                   # Streamlit web app entry point
├── config.py                # Settings and constants (tickers, risk parameters)
├── data.py                  # Fetch and preprocess stock data
├── optimization.py          # Portfolio optimization math (MPT algorithms)
├── visualization.py         # Matplotlib charts (CLI)
├── visualization_plotly.py  # Plotly charts (web app)
├── main.py                  # CLI entry point
└── output/                  # Generated files (charts, CSV)
requirements.txt             # Dependencies for Streamlit Cloud
```

## Setup

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
brew install uv  # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS

# Install dependencies
uv sync
```

## Usage

### Web App (Recommended)

```bash
uv run streamlit run src/app.py
```

This launches an interactive web interface where you can:
- Select stocks from a dropdown
- Adjust date range and weight constraints
- View interactive Plotly charts
- Download results as CSV

### CLI

```bash
uv run python src/main.py
```

Edit `src/config.py` to configure settings for CLI mode.

## Configuration
Edit settings in `src/config.py`:
- `TICKERS`: List of stock symbols
- `RISK_FREE_RATE`: Risk-free rate (default: 0.02)
- `MAX_WEIGHT`: Maximum weight per asset (default: 0.30)
- `MIN_WEIGHT`: Minimum weight per asset (default: 0.05)
- `USE_MAX_CONSTRAINT`: Enable max weight constraint (default: True)
- `USE_MIN_CONSTRAINT`: Enable min weight constraint (default: False)
- `START_DATE` / `END_DATE`: Date range for historical data

## Output (CLI)
Results are saved to `src/output/`:
- `optimal_portfolio_weights.csv`
- `efficient_frontier.png`
- `correlation_matrix.png`
- `price_history.png`
- `portfolio_allocation.png`
- `risk_return_comparison.png`
- `sharpe_comparison.png`

## Deployment

Deploy to Streamlit Cloud for free:

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `src/app.py`
5. Deploy

The app will be available at `https://your-app-name.streamlit.app`

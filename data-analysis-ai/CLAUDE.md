# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professional cryptocurrency data analysis pipeline with AI-powered insights built using Python. The project demonstrates data science best practices with a clean, reproducible architecture focused on crypto market analysis.

## Python Environment

**CRITICAL**: Always use `uv` for Python tasks:
- `uv run <file.py>` instead of `python <file.py>`
- `uv run pytest` instead of `pytest`
- `uv pip install` instead of `pip install`

The project uses Python 3.11+ with dependencies managed via `pyproject.toml`.

## Core Commands

### Pipeline Execution
```bash
# Run complete analysis pipeline
uv run run_pipeline.py

# Run specific pipeline steps
uv run run_pipeline.py --steps collect crypto top100
uv run run_pipeline.py --steps report

# Run with custom seed for reproducibility
uv run run_pipeline.py --seed 42
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_analysis.py

# Run tests with verbose output
uv run pytest tests/ -v
```

### Development
```bash
# Run main entry point
uv run main.py

# Install dependencies
uv pip install -e .
```

## Architecture

### Pipeline Design
The project follows a master pipeline pattern orchestrated by `run_pipeline.py`:

1. **Data Collection** (`coingecko_api.py`) - Fetches cryptocurrency data
2. **Crypto Analysis** (`crypto_ai_analysis.py`) - Technical analysis with AI insights
3. **Top 100 Analysis** (`top100_crypto_analysis.py`) - Market-wide analysis
4. **Report Generation** - Automated markdown reports

### Key Components

- **Configuration Management**: Centralized in `configs/config.py` with reproducible settings
- **Data Flow**: `data/raw/` → `src/` processing → `results/data/` and `results/figures/`
- **Logging**: Complete pipeline execution tracking to `analysis.log`
- **Reproducibility**: Fixed random seeds (default: 42) and version-pinned dependencies

### Module Structure
- `src/crypto_ai_analysis.py` - Main `CryptoAIAnalyzer` class with technical indicators
- `src/coingecko_api.py` - API integration for crypto data
- `src/top100_*.py` - Market analysis modules
- `configs/config.py` - All configuration constants and directory setup

### Path Management
The project uses `sys.path.insert(0, str(Path(__file__).parent / "src"))` pattern to ensure modules are importable. All paths are managed through `configs/config.py` using `pathlib.Path`.

## Configuration

### Analysis Parameters
Edit `configs/config.py` to modify:
- `ANALYSIS_CONFIG["crypto"]["default_symbols"]` - Cryptocurrencies to analyze
- `ANALYSIS_CONFIG["crypto"]["default_days"]` - Analysis timeframe
- `ANALYSIS_CONFIG["indicators"]` - Technical indicator parameters
- `ANALYSIS_CONFIG["random_seed"]` - For reproducibility

### Environment Variables
- `COINGECKO_API_KEY` - Optional CoinGecko API key for higher rate limits

## Output Structure

After pipeline execution:
- `results/data/` - CSV files with analysis results
- `results/figures/` - Generated charts and visualizations
- `analysis.log` - Complete execution log
- `analysis_report.md` - Generated summary report

## Key Classes

- `CryptoAIAnalyzer` - Main analysis engine with methods:
  - `fetch_crypto_data()` - Data retrieval
  - `calculate_indicators()` - Technical analysis
  - `ai_trading_recommendation()` - AI-powered signals
  - `create_advanced_dashboard()` - Visualization generation

## Development Workflow

1. Modify source files in `src/`
2. Test with `uv run pytest tests/`
3. Run pipeline with `uv run run_pipeline.py`
4. Check outputs in `results/`
5. Review logs in `analysis.log`

## Recent Improvements

### Error Handling & Robustness
- Added `tenacity` for API retry logic with exponential backoff
- Enhanced error handling in `coingecko_api.py` with comprehensive logging
- Added `pydantic` data validation models for crypto data integrity

### Advanced Technical Analysis
- **Bollinger Bands**: Upper/lower bands for overbought/oversold detection
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Enhanced Signals**: Multi-indicator confirmation system (MA + RSI + BB + MACD)
- **Signal Types**: HOLD, BUY, STRONG_BUY, SELL with sophisticated logic

### Interactive Visualizations
- **Plotly Integration**: Interactive HTML dashboards with hover details
- **Multi-panel Layout**: Price/MA, RSI, Volume in synchronized subplots
- **Dual Output**: Both static (matplotlib) and interactive (plotly) charts

### Testing & Validation
- **Edge Case Testing**: API failures, empty datasets, null values, extreme prices
- **Data Validation**: Automatic cleaning of negative values and outliers
- **Mock Testing**: Comprehensive test scenarios with unittest.mock

### API Resilience
- **Retry Logic**: 3 attempts with exponential backoff for failed requests
- **Rate Limiting**: Respectful API usage with 1-second delays
- **Timeout Handling**: 30-second timeouts to prevent hanging requests
- **JSON Validation**: Proper error handling for malformed API responses

## Dependencies

Core stack: pandas, numpy, yfinance, matplotlib, seaborn, plotly, scikit-learn, xgboost
Error handling: tenacity, pydantic, requests
Development: jupyter, pytest
All managed via `pyproject.toml` with `uv`.
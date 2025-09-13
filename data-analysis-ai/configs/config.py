"""
Configuration file for Data Analysis AI Project
Centralized settings for reproducibility
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_OUTPUT_DIR = RESULTS_DIR / "data"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, DATA_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Analysis parameters
ANALYSIS_CONFIG = {
    # Random seed for reproducibility
    "random_seed": 42,

    # Crypto analysis settings
    "crypto": {
        "default_symbols": ["BTC-USD", "ETH-USD"],
        "default_days": 30,
        "top_n_cryptos": 100,
    },

    # Technical indicators
    "indicators": {
        "ma_short": 7,
        "ma_long": 21,
        "rsi_period": 14,
        "volatility_window": 7,
    },

    # Visualization settings
    "plotting": {
        "figure_size": (12, 8),
        "dpi": 100,
        "style": "seaborn-v0_8",
    }
}

# API Configuration (use environment variables for security)
API_CONFIG = {
    "coingecko": {
        "base_url": "https://api.coingecko.com/api/v3",
        "api_key": os.getenv("COINGECKO_API_KEY", ""),
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "analysis.log"
}

# File paths for data
DATA_FILES = {
    "crypto_data": RAW_DATA_DIR / "crypto_data.csv",
    "processed_crypto": PROCESSED_DATA_DIR / "processed_crypto.csv",
    "analysis_results": DATA_OUTPUT_DIR / "analysis_results.csv",
}

def get_config():
    """Return the complete configuration dictionary"""
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "src": str(SRC_DIR),
            "data": str(DATA_DIR),
            "results": str(RESULTS_DIR),
        },
        "analysis": ANALYSIS_CONFIG,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "files": {k: str(v) for k, v in DATA_FILES.items()},
    }
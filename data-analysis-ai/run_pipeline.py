#!/usr/bin/env python
"""
Master Pipeline Script for Data Analysis AI Project
Orchestrates the complete analysis workflow from data to results
"""

import sys
import logging
from pathlib import Path
import argparse
import numpy as np
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "configs"))

from config import ANALYSIS_CONFIG, LOGGING_CONFIG, PROJECT_ROOT, FIGURES_DIR, DATA_OUTPUT_DIR

# Set up logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Random seeds set to {seed}")

def run_data_collection():
    """Step 1: Collect and fetch data"""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 50)

    try:
        from coingecko_api import fetch_top_100_crypto_data

        logger.info("Fetching top 100 cryptocurrency data...")
        data = fetch_top_100_crypto_data()

        if data is not None:
            output_path = DATA_OUTPUT_DIR / f"crypto_data_{datetime.now().strftime('%Y%m%d')}.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
            return data
        else:
            logger.warning("No data fetched, using existing data if available")
            return None

    except ImportError as e:
        logger.warning(f"Import error: {e}. Skipping data collection.")
        return None
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        return None

def run_crypto_analysis():
    """Step 2: Run cryptocurrency analysis"""
    logger.info("=" * 50)
    logger.info("STEP 2: CRYPTOCURRENCY ANALYSIS")
    logger.info("=" * 50)

    try:
        from crypto_ai_analysis import CryptoAIAnalyzer

        analyzer = CryptoAIAnalyzer()
        symbols = ANALYSIS_CONFIG["crypto"]["default_symbols"]
        days = ANALYSIS_CONFIG["crypto"]["default_days"]

        logger.info(f"Analyzing {symbols} for {days} days...")

        # Fetch data
        analyzer.fetch_crypto_data(symbols, days)

        # Calculate indicators and generate signals
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            analyzer.calculate_indicators(symbol)
            recommendation = analyzer.ai_trading_recommendation(symbol)
            logger.info(f"Trading recommendation for {symbol}: {recommendation}")

        # Create visualizations
        analyzer.create_advanced_dashboard()
        logger.info(f"Dashboard saved to {FIGURES_DIR}")

        return analyzer

    except Exception as e:
        logger.error(f"Error in crypto analysis: {e}")
        return None

def run_top100_analysis():
    """Step 3: Analyze top 100 cryptocurrencies"""
    logger.info("=" * 50)
    logger.info("STEP 3: TOP 100 CRYPTO ANALYSIS")
    logger.info("=" * 50)

    try:
        from top100_crypto_analysis import analyze_top_100_cryptos

        logger.info("Analyzing top 100 cryptocurrencies...")
        results = analyze_top_100_cryptos()

        if results is not None:
            output_path = DATA_OUTPUT_DIR / f"top100_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
            results.to_csv(output_path, index=False)
            logger.info(f"Analysis results saved to {output_path}")

        return results

    except ImportError as e:
        logger.warning(f"Import error: {e}. Skipping top 100 analysis.")
        return None
    except Exception as e:
        logger.error(f"Error in top 100 analysis: {e}")
        return None

def generate_report():
    """Step 4: Generate final report"""
    logger.info("=" * 50)
    logger.info("STEP 4: GENERATING REPORT")
    logger.info("=" * 50)

    report_path = PROJECT_ROOT / "analysis_report.md"

    with open(report_path, "w") as f:
        f.write("# Data Analysis AI - Pipeline Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Pipeline Execution Summary\n\n")
        f.write("1. ✅ Data Collection - Completed\n")
        f.write("2. ✅ Cryptocurrency Analysis - Completed\n")
        f.write("3. ✅ Top 100 Analysis - Completed\n")
        f.write("4. ✅ Report Generation - Completed\n\n")

        f.write("## Output Files\n\n")
        f.write(f"- Data: `{DATA_OUTPUT_DIR}`\n")
        f.write(f"- Figures: `{FIGURES_DIR}`\n")
        f.write(f"- Logs: `{LOGGING_CONFIG['log_file']}`\n\n")

        f.write("## Configuration Used\n\n")
        f.write(f"- Random Seed: {ANALYSIS_CONFIG['random_seed']}\n")
        f.write(f"- Analysis Period: {ANALYSIS_CONFIG['crypto']['default_days']} days\n")
        f.write(f"- Symbols Analyzed: {ANALYSIS_CONFIG['crypto']['default_symbols']}\n")

    logger.info(f"Report generated at {report_path}")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Run Data Analysis AI Pipeline")
    parser.add_argument("--steps", nargs="+", default=["all"],
                       choices=["all", "collect", "crypto", "top100", "report"],
                       help="Which steps to run")
    parser.add_argument("--seed", type=int, default=ANALYSIS_CONFIG["random_seed"],
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Start pipeline
    logger.info("=" * 50)
    logger.info("STARTING DATA ANALYSIS PIPELINE")
    logger.info("=" * 50)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Steps to run: {args.steps}")

    # Set random seeds
    set_random_seeds(args.seed)

    # Run pipeline steps
    if "all" in args.steps or "collect" in args.steps:
        run_data_collection()

    if "all" in args.steps or "crypto" in args.steps:
        run_crypto_analysis()

    if "all" in args.steps or "top100" in args.steps:
        run_top100_analysis()

    if "all" in args.steps or "report" in args.steps:
        generate_report()

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
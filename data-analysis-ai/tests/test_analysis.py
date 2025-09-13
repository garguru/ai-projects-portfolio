"""
Basic tests for Data Analysis AI project
Demonstrates testing best practices
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "configs"))

from config import ANALYSIS_CONFIG, PROJECT_ROOT


class TestConfiguration(unittest.TestCase):
    """Test configuration settings"""

    def test_directories_exist(self):
        """Test that all required directories exist"""
        dirs_to_check = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "results",
            PROJECT_ROOT / "configs",
        ]
        for directory in dirs_to_check:
            self.assertTrue(directory.exists(), f"Directory {directory} does not exist")

    def test_random_seed_set(self):
        """Test that random seed is properly configured"""
        self.assertIsNotNone(ANALYSIS_CONFIG.get("random_seed"))
        self.assertEqual(ANALYSIS_CONFIG["random_seed"], 42)

    def test_config_structure(self):
        """Test configuration has required keys"""
        required_keys = ["random_seed", "crypto", "indicators", "plotting"]
        for key in required_keys:
            self.assertIn(key, ANALYSIS_CONFIG, f"Missing config key: {key}")


class TestDataValidation(unittest.TestCase):
    """Test data validation functions"""

    def test_validate_crypto_symbols(self):
        """Test cryptocurrency symbol validation"""
        valid_symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
        invalid_symbols = ["INVALID", "", None, 123]

        for symbol in valid_symbols:
            self.assertTrue(self._is_valid_symbol(symbol))

        for symbol in invalid_symbols:
            self.assertFalse(self._is_valid_symbol(symbol))

    def _is_valid_symbol(self, symbol):
        """Helper: Check if symbol is valid"""
        if not symbol or not isinstance(symbol, str):
            return False
        parts = symbol.split("-")
        return len(parts) == 2 and len(parts[0]) > 0 and parts[1] in ["USD", "USDT", "BTC", "ETH"]

    def test_data_quality_checks(self):
        """Test data quality validation"""
        # Create sample data
        good_data = pd.DataFrame({
            "price": [100, 101, 102, 103],
            "volume": [1000, 1100, 1200, 1300],
            "timestamp": pd.date_range("2024-01-01", periods=4)
        })

        bad_data = pd.DataFrame({
            "price": [100, None, 102, -5],  # Has null and negative
            "volume": [1000, 1100, None, 1300],
            "timestamp": pd.date_range("2024-01-01", periods=4)
        })

        self.assertTrue(self._validate_data(good_data))
        self.assertFalse(self._validate_data(bad_data))

    def _validate_data(self, df):
        """Helper: Validate dataframe quality"""
        if df.isnull().any().any():
            return False
        if (df.select_dtypes(include=[np.number]) < 0).any().any():
            return False
        return True


class TestAnalysisFunctions(unittest.TestCase):
    """Test core analysis functions"""

    def test_moving_average_calculation(self):
        """Test moving average calculation"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma3 = data.rolling(window=3).mean()

        # Check first values are NaN
        self.assertTrue(pd.isna(ma3.iloc[0]))
        self.assertTrue(pd.isna(ma3.iloc[1]))

        # Check calculated values
        self.assertEqual(ma3.iloc[2], 2.0)  # (1+2+3)/3
        self.assertEqual(ma3.iloc[3], 3.0)  # (2+3+4)/3

    def test_rsi_calculation(self):
        """Test RSI calculation logic"""
        prices = pd.Series([44, 44.25, 44.5, 43.75, 44.75, 45.5, 45.25, 46, 47, 46.75])
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()

        # Basic checks
        self.assertGreaterEqual(gain.min(), 0)
        self.assertGreaterEqual(loss.min(), 0)

    def test_volatility_calculation(self):
        """Test volatility calculation"""
        prices = pd.Series([100, 102, 98, 101, 103, 99, 102])
        returns = prices.pct_change()
        volatility = returns.std()

        self.assertGreater(volatility, 0)
        self.assertLess(volatility, 1)  # Should be reasonable percentage


class TestPipeline(unittest.TestCase):
    """Test pipeline functionality"""

    def test_pipeline_imports(self):
        """Test that pipeline script can import required modules"""
        try:
            import run_pipeline
            self.assertTrue(hasattr(run_pipeline, 'main'))
            self.assertTrue(hasattr(run_pipeline, 'set_random_seeds'))
        except ImportError as e:
            self.fail(f"Failed to import pipeline: {e}")

    def test_random_seed_reproducibility(self):
        """Test that setting seed produces reproducible results"""
        np.random.seed(42)
        first_random = [np.random.random() for _ in range(5)]

        np.random.seed(42)
        second_random = [np.random.random() for _ in range(5)]

        self.assertEqual(first_random, second_random)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    print("=" * 50)
    print("Running Data Analysis AI Tests")
    print("=" * 50)
    run_tests()
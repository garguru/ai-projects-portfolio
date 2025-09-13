"""
Unit tests for I/O utilities
Micro-tests demonstrating testing best practices
"""

import unittest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from common.io import load_csv, save_csv, validate_dataframe, load_sample_data


class TestIOUtilities(unittest.TestCase):
    """Test data I/O functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create sample data
        self.sample_data = pd.DataFrame({
            'price': [100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300],
            'date': pd.date_range('2024-01-01', periods=4)
        })

    def test_load_sample_data(self):
        """Test sample data generation"""
        df = load_sample_data()

        # Basic structure tests
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('price', df.columns)
        self.assertIn('date', df.columns)
        self.assertIn('volume', df.columns)
        self.assertIn('symbol', df.columns)

        # Data quality tests
        self.assertGreater(len(df), 0)
        self.assertTrue(df['price'].dtype in ['float64', 'int64'])
        self.assertTrue(df['volume'].dtype in ['float64', 'int64'])

    def test_save_and_load_csv(self):
        """Test CSV save/load round trip"""
        test_file = self.temp_dir / "test_data.csv"

        # Save data
        saved_path = save_csv(self.sample_data, test_file)
        self.assertEqual(saved_path, test_file)
        self.assertTrue(test_file.exists())

        # Load data back
        loaded_data = load_csv(test_file)

        # Verify data integrity
        self.assertEqual(len(loaded_data), len(self.sample_data))
        self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))

    def test_validate_dataframe(self):
        """Test dataframe validation"""
        # Test valid data
        result = validate_dataframe(self.sample_data, required_columns=['price', 'volume'])
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['metrics']['rows'], 4)

        # Test missing required columns
        result = validate_dataframe(self.sample_data, required_columns=['missing_col'])
        self.assertFalse(result['is_valid'])
        self.assertTrue(any('Missing required columns' in issue for issue in result['issues']))

        # Test empty dataframe
        empty_df = pd.DataFrame()
        result = validate_dataframe(empty_df)
        self.assertFalse(result['is_valid'])

    def test_load_nonexistent_file(self):
        """Test error handling for missing files"""
        with self.assertRaises(FileNotFoundError):
            load_csv(self.temp_dir / "nonexistent.csv")

    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("🧪 Running I/O utility tests...")
    unittest.main(verbosity=2)
"""
Data I/O utilities for the Data Analysis AI project
Clean, reusable functions for loading and saving data
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load CSV file with error handling and validation

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, **kwargs)

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        logger.info(f"Loaded CSV: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
        return df

    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file contains no data: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {e}")

def save_csv(df: pd.DataFrame, file_path: Union[str, Path],
             create_dirs: bool = True, **kwargs) -> Path:
    """
    Save DataFrame to CSV with directory creation

    Args:
        df: DataFrame to save
        file_path: Output file path
        create_dirs: Whether to create parent directories
        **kwargs: Additional arguments passed to df.to_csv

    Returns:
        Path object of saved file
    """
    file_path = Path(file_path)

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Default arguments for cleaner CSV output
    csv_kwargs = {'index': False}
    csv_kwargs.update(kwargs)

    df.to_csv(file_path, **csv_kwargs)
    logger.info(f"Saved CSV: {file_path} ({len(df)} rows)")

    return file_path

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate DataFrame and return quality metrics

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Dictionary with validation results and quality metrics
    """
    validation = {
        'is_valid': True,
        'issues': [],
        'metrics': {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    }

    # Check if DataFrame is empty
    if df.empty:
        validation['is_valid'] = False
        validation['issues'].append("DataFrame is empty")
        return validation

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation['is_valid'] = False
            validation['issues'].append(f"Missing required columns: {missing_cols}")

    # Check for excessive missing data
    missing_pct = validation['metrics']['missing_values'] / (len(df) * len(df.columns))
    if missing_pct > 0.5:
        validation['is_valid'] = False
        validation['issues'].append(f"Too many missing values: {missing_pct:.1%}")

    # Check for suspicious data types
    object_cols = df.select_dtypes(include='object').columns
    if len(object_cols) > len(df.columns) * 0.8:
        validation['issues'].append("Most columns are object type - check data parsing")

    logger.info(f"Data validation: {'✅ Valid' if validation['is_valid'] else '❌ Issues found'}")

    return validation

def load_sample_data() -> pd.DataFrame:
    """
    Load sample cryptocurrency data for testing and demos

    Returns:
        Sample DataFrame with crypto data
    """
    from datetime import datetime, timedelta
    import numpy as np

    # Generate sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30),
                         end=datetime.now(), freq='D')

    np.random.seed(42)  # Reproducible sample
    prices = [45000]  # Starting BTC price

    for _ in range(len(dates) - 1):
        change = np.random.normal(0.001, 0.03)
        prices.append(prices[-1] * (1 + change))

    sample_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.lognormal(15, 0.5, len(dates)),
        'symbol': 'BTC-USD'
    })

    logger.info("Generated sample cryptocurrency data")
    return sample_data

def get_data_info(df: pd.DataFrame) -> str:
    """
    Get formatted string with DataFrame information

    Args:
        df: DataFrame to analyze

    Returns:
        Formatted string with data summary
    """
    info = f"""
📊 Dataset Information:
  Shape: {df.shape}
  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
  Missing values: {df.isnull().sum().sum()}

📈 Numeric columns: {len(df.select_dtypes(include='number').columns)}
📝 Text columns: {len(df.select_dtypes(include='object').columns)}
📅 Date columns: {len(df.select_dtypes(include='datetime').columns)}
"""
    return info
"""
Test Module for Data Loader

This module contains unit tests for the data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_price_data, validate_data

@pytest.fixture
def sample_data_path():
    """Fixture to provide the path to sample data."""
    return "data/sample_data.csv"

def test_load_price_data(sample_data_path):
    """Test loading price data from CSV file."""
    # Load the data
    df = load_price_data(sample_data_path)
    
    # Check if DataFrame is not empty
    assert not df.empty
    
    # Check if date is the index
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Check if all required columns are present
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_columns)
    
    # Check if data types are correct
    assert df['open'].dtype in [np.float64, np.int64]
    assert df['high'].dtype in [np.float64, np.int64]
    assert df['low'].dtype in [np.float64, np.int64]
    assert df['close'].dtype in [np.float64, np.int64]
    assert df['volume'].dtype in [np.float64, np.int64]

def test_load_price_data_file_not_found():
    """Test loading price data with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_price_data("non_existent_file.csv")

def test_validate_data():
    """Test data validation function."""
    # Create valid data
    valid_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 1200, 1100, 1300, 1400]
    })
    
    # Test valid data
    assert validate_data(valid_data) is True
    
    # Test missing column
    invalid_data = valid_data.drop('close', axis=1)
    with pytest.raises(ValueError):
        validate_data(invalid_data)
    
    # Test negative prices
    invalid_data = valid_data.copy()
    invalid_data.loc[0, 'open'] = -100
    with pytest.raises(ValueError):
        validate_data(invalid_data)
    
    # Test high < low
    invalid_data = valid_data.copy()
    invalid_data.loc[0, 'high'] = 98
    with pytest.raises(ValueError):
        validate_data(invalid_data) 
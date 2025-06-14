"""
Data Loading Module

This module handles loading and basic preprocessing of price data for the backtester.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional

# Required columns for price data
REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

def load_price_data(file_path: str) -> pd.DataFrame:
    """
    Load price data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing price data
        
    Returns:
        pd.DataFrame: DataFrame containing the price data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the required columns are not present in the data
    """
    # Convert string path to Path object
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Price data file not found: {file_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(path)
        
        # Validate the data
        validate_data(df)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"Price data file is empty: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Error parsing price data file: {file_path}")

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded price data.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if data is valid, False otherwise
        
    Raises:
        ValueError: If data validation fails
    """
    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("DataFrame contains missing values")
    
    # Check for non-numeric values in price columns
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            raise ValueError(f"Column '{col}' contains non-numeric values")
    
    # Check for negative prices
    if (df[price_columns] < 0).any().any():
        raise ValueError("DataFrame contains negative prices")
    
    # Check for high > low
    if not (df['high'] >= df['low']).all():
        raise ValueError("High prices are not always greater than or equal to low prices")
    
    return True 
"""
Test Module for Strategy

This module contains unit tests for the strategy implementation.
"""

import pytest
import pandas as pd
import numpy as np
from src.strategy import calculate_sma, calculate_ema, generate_signals

def test_calculate_sma():
    """Test Simple Moving Average calculation."""
    prices = pd.Series([1, 2, 3, 4, 5])
    period = 3
    expected = pd.Series([1.0, 1.5, 2.0, 3.0, 4.0])
    result = calculate_sma(prices, period)
    pd.testing.assert_series_equal(result, expected)

def test_calculate_ema():
    """Test Exponential Moving Average calculation."""
    prices = pd.Series([1, 2, 3, 4, 5])
    period = 3
    # Calculated manually or with pandas for reference
    expected = prices.ewm(span=period, adjust=False).mean()
    result = calculate_ema(prices, period)
    pd.testing.assert_series_equal(result, expected)

def test_generate_signals():
    """Test signal generation logic."""
    # Fast MA crosses above slow MA at index 2, below at index 4
    fast_ma = pd.Series([1, 2, 3, 2, 1])
    slow_ma = pd.Series([2, 2, 2, 2, 2])
    # At index 2: fast crosses above slow (buy)
    # At index 4: fast crosses below slow (sell)
    expected = pd.Series([0, 0, 1, 0, -1])
    result = generate_signals(fast_ma, slow_ma)
    pd.testing.assert_series_equal(result, expected) 
"""
Strategy Module

This module implements the Moving Average Crossover strategy and related calculations.
"""

import pandas as pd
import numpy as np

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices (pd.Series): Price series
        period (int): Moving average period
        
    Returns:
        pd.Series: Simple Moving Average series
        
    Raises:
        ValueError: If period is less than 1
    """
    if period < 1:
        raise ValueError("Moving average period must be at least 1")
    return prices.rolling(window=period, min_periods=1).mean()

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices (pd.Series): Price series
        period (int): Moving average period
        
    Returns:
        pd.Series: Exponential Moving Average series
        
    Raises:
        ValueError: If period is less than 1
    """
    if period < 1:
        raise ValueError("Moving average period must be at least 1")
    return prices.ewm(span=period, adjust=False).mean()

def generate_signals(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
    """
    Generate trading signals based on MA crossovers.
    
    Args:
        fast_ma (pd.Series): Fast moving average series
        slow_ma (pd.Series): Slow moving average series
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        
    Raises:
        ValueError: If input series have different lengths or indices
    """
    if len(fast_ma) != len(slow_ma):
        raise ValueError("Fast and slow MA series must have the same length")
    if not fast_ma.index.equals(slow_ma.index):
        raise ValueError("Fast and slow MA series must have the same index")
    
    diff = fast_ma - slow_ma
    prev_diff = diff.shift(1)
    signals = pd.Series(0, index=fast_ma.index)
    # Buy signal: previous diff <= 0 and current diff > 0
    signals[(prev_diff <= 0) & (diff > 0)] = 1
    # Sell signal: previous diff >= 0 and current diff < 0
    signals[(prev_diff >= 0) & (diff < 0)] = -1
    return signals 
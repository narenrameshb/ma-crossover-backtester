"""
Integration Test Module

This module contains integration tests for the entire backtesting system.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_price_data
from src.strategy import calculate_sma, calculate_ema, generate_signals
from src.backtester import Backtester

def test_full_system():
    """Test the entire backtesting system from data loading to results."""
    # Load sample data
    data_path = "data/sample_data.csv"
    df = load_price_data(data_path)
    prices = df['close']
    
    # Calculate moving averages
    fast_ma = calculate_sma(prices, period=5)
    slow_ma = calculate_sma(prices, period=10)
    
    # Generate signals
    signals = generate_signals(fast_ma, slow_ma)
    
    # Run backtest
    bt = Backtester(initial_capital=100000.0)
    results = bt.run(prices, signals)
    
    # Calculate metrics
    metrics = bt.calculate_metrics()
    
    # Verify results
    assert isinstance(results['portfolio_value'], pd.Series)
    assert isinstance(results['positions'], pd.Series)
    assert isinstance(results['trades'], list)
    assert len(results['trades']) > 0
    
    # Verify metrics
    assert 'final_value' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'total_trades' in metrics
    
    # Verify data types and ranges
    assert isinstance(metrics['final_value'], (int, float))
    assert isinstance(metrics['sharpe_ratio'], (int, float))
    assert isinstance(metrics['max_drawdown'], (int, float))
    assert isinstance(metrics['win_rate'], (int, float))
    assert isinstance(metrics['total_trades'], int)
    
    assert metrics['max_drawdown'] <= 0  # Drawdown should be negative
    assert 0 <= metrics['win_rate'] <= 1  # Win rate should be between 0 and 1
    assert metrics['total_trades'] >= 0  # Total trades should be non-negative

def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with empty data
    with pytest.raises(ValueError):
        load_price_data("data/empty_data.csv")
    
    # Test with invalid data format
    with pytest.raises(ValueError):
        load_price_data("data/invalid_data.csv")
    
    # Test with single price point
    single_price = pd.Series([100], index=pd.date_range('2023-01-01', periods=1))
    fast_ma = calculate_sma(single_price, period=5)
    assert len(fast_ma) == 1
    
    # Test with all zeros
    zero_prices = pd.Series([0, 0, 0], index=pd.date_range('2023-01-01', periods=3))
    fast_ma = calculate_sma(zero_prices, period=2)
    assert (fast_ma == 0).all()
    
    # Test with very large numbers
    large_prices = pd.Series([1e9, 1e9, 1e9], index=pd.date_range('2023-01-01', periods=3))
    fast_ma = calculate_sma(large_prices, period=2)
    assert (fast_ma == 1e9).all()

def test_input_validation():
    """Test input validation for various parameters."""
    # Test invalid MA periods
    prices = pd.Series([100, 101, 102], index=pd.date_range('2023-01-01', periods=3))
    with pytest.raises(ValueError):
        calculate_sma(prices, period=0)
    with pytest.raises(ValueError):
        calculate_sma(prices, period=-1)
    
    # Test invalid initial capital
    with pytest.raises(ValueError):
        Backtester(initial_capital=-1000)
    
    # Test invalid risk-free rate
    bt = Backtester(initial_capital=100000.0)
    with pytest.raises(ValueError):
        bt.calculate_metrics(risk_free_rate=-0.1) 
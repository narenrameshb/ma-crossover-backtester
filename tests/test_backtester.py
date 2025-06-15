"""
Test Module for Backtester

This module contains unit tests for the backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtester import Backtester

def test_backtester_run():
    """Test basic backtesting logic."""
    prices = pd.Series([10, 12, 14, 13, 11],
                       index=pd.date_range('2023-01-01', periods=5))
    signals = pd.Series([0, 1, 0, -1, 0],
                        index=prices.index)
    bt = Backtester(initial_capital=100)
    results = bt.run(prices, signals)
    # After buy on day 2 at price 12, should hold 8 shares, cash = 4
    # After sell on day 4 at price 13, cash = 4 + 8*13 = 108
    assert results['positions'].iloc[1] == 1  # Long after buy
    assert results['positions'].iloc[3] == 0  # Flat after sell
    assert results['trades'][0]['type'] == 'buy'
    assert results['trades'][1]['type'] == 'sell'
    assert results['portfolio_value'].iloc[-1] == 108  # Final portfolio value

def test_backtester_metrics():
    """Test performance metrics calculation."""
    prices = pd.Series([10, 12, 14, 13, 11],
                       index=pd.date_range('2023-01-01', periods=5))
    signals = pd.Series([0, 1, 0, -1, 0],
                        index=prices.index)
    bt = Backtester(initial_capital=100)
    bt.run(prices, signals)
    metrics = bt.calculate_metrics()
    # There is one round-trip trade, and it is profitable
    assert metrics['win_rate'] == 1.0
    # Final value should match previous test
    assert metrics['final_value'] == 108
    # Max drawdown should be <= 0
    assert metrics['max_drawdown'] <= 0
    # Sharpe ratio should be a finite number
    assert np.isfinite(metrics['sharpe_ratio']) 
"""
Utility Module

This module contains utility functions for plotting and visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_price_and_ma(prices: pd.Series, fast_ma: pd.Series, slow_ma: pd.Series):
    """
    Plot price with fast and slow moving averages.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label='Price', color='black')
    plt.plot(fast_ma, label='Fast MA', color='blue')
    plt.plot(slow_ma, label='Slow MA', color='red')
    plt.title('Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_signals(prices: pd.Series, signals: pd.Series):
    """
    Plot buy/sell signals on the price chart.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label='Price', color='black')
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    plt.scatter(buy_signals, prices.loc[buy_signals], marker='^', color='green', label='Buy', zorder=5)
    plt.scatter(sell_signals, prices.loc[sell_signals], marker='v', color='red', label='Sell', zorder=5)
    plt.title('Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_portfolio_value(portfolio_value: pd.Series):
    """
    Plot the portfolio value (equity curve) over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value, label='Portfolio Value', color='purple')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.show() 
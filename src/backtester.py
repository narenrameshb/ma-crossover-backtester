"""
Backtesting Module

This module implements the backtesting engine for the MA Crossover strategy.
"""

import pandas as pd
import numpy as np

class Backtester:
    """
    Backtesting engine for the MA Crossover strategy.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Initial capital for the backtest
            
        Raises:
            ValueError: If initial_capital is less than or equal to 0
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than 0")
        self.initial_capital = initial_capital
        self.positions = None
        self.portfolio_value = None
        self.trades = []
        
    def run(self, prices: pd.Series, signals: pd.Series) -> dict:
        """
        Run the backtest.
        
        Args:
            prices (pd.Series): Price series
            signals (pd.Series): Trading signals
            
        Returns:
            dict: Dictionary containing backtest results
            
        Raises:
            ValueError: If input series have different lengths or indices
        """
        if len(prices) != len(signals):
            raise ValueError("Price and signal series must have the same length")
        if not prices.index.equals(signals.index):
            raise ValueError("Price and signal series must have the same index")
        
        # Initialize
        position = 0  # 0 = flat, 1 = long
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        positions = []
        
        for date, price in prices.items():
            signal = signals.loc[date]
            # Buy signal
            if signal == 1 and position == 0:
                shares = cash // price
                cash -= shares * price
                position = 1
                self.trades.append({'date': date, 'type': 'buy', 'price': price, 'shares': shares})
            # Sell signal
            elif signal == -1 and position == 1:
                cash += shares * price
                position = 0
                self.trades.append({'date': date, 'type': 'sell', 'price': price, 'shares': shares})
                shares = 0
            # Record position
            positions.append(position)
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
        
        self.positions = pd.Series(positions, index=prices.index)
        self.portfolio_value = pd.Series(portfolio_values, index=prices.index)
        
        return {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions,
            'trades': self.trades
        }
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> dict:
        """
        Calculate performance metrics.
        
        Args:
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation (annualized, as decimal)
            
        Returns:
            dict: Dictionary containing performance metrics
            
        Raises:
            ValueError: If backtest hasn't been run or risk_free_rate is negative
        """
        if self.portfolio_value is None or not self.trades:
            raise ValueError("Backtest must be run before calculating metrics")
        if risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
        
        # Calculate returns
        returns = self.portfolio_value.pct_change().fillna(0)
        
        # Sharpe ratio (assume daily returns, 252 trading days)
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-9)
        
        # Maximum drawdown
        cumulative_max = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Win rate (fraction of profitable trades)
        wins = 0
        total_trades = 0
        for i in range(1, len(self.trades), 2):  # look for sell after buy
            buy = self.trades[i-1]
            sell = self.trades[i]
            if buy['type'] == 'buy' and sell['type'] == 'sell':
                profit = (sell['price'] - buy['price']) * buy['shares']
                if profit > 0:
                    wins += 1
                total_trades += 1
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        return {
            'final_value': self.portfolio_value.iloc[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        } 
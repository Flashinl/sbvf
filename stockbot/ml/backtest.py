"""
Backtesting Framework for Multi-Horizon Trading Signals

Simulates trading strategy based on model predictions and compares to buy-and-hold baseline

Features:
- Transaction costs
- Position sizing
- Portfolio tracking
- Performance metrics (Sharpe, max drawdown, win rate)
- Comparison vs buy-and-hold
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    position_size: float = 0.1  # Use 10% of capital per position
    min_confidence: float = 0.6  # Minimum confidence to take a position
    stop_loss: float = -0.10  # -10% stop loss
    take_profit: float = 0.20  # +20% take profit
    max_positions: int = 10  # Maximum concurrent positions


@dataclass
class Trade:
    """Represents a single trade"""
    ticker: str
    entry_date: datetime
    entry_price: float
    signal: str  # 'BUY' or 'SELL'
    quantity: int
    confidence: float
    horizon: str

    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'target', 'stop_loss', 'signal_change'

    def pnl(self) -> float:
        """Calculate profit/loss"""
        if self.exit_price is None:
            return 0.0

        if self.signal == 'BUY':
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity

    def return_pct(self) -> float:
        """Calculate return percentage"""
        if self.exit_price is None:
            return 0.0

        if self.signal == 'BUY':
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100


class Portfolio:
    """Manages portfolio state during backtesting"""

    def __init__(self, initial_capital: float, transaction_cost: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost = transaction_cost
        self.positions: Dict[str, Trade] = {}  # Active positions
        self.closed_trades: List[Trade] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []

    def can_open_position(self, cost: float, max_positions: int) -> bool:
        """Check if we can open a new position"""
        return (
            self.cash >= cost and
            len(self.positions) < max_positions
        )

    def open_position(self, trade: Trade) -> bool:
        """Open a new position"""
        cost = trade.entry_price * trade.quantity
        total_cost = cost * (1 + self.transaction_cost)

        if not self.can_open_position(total_cost, max_positions=999):
            return False

        self.cash -= total_cost
        self.positions[trade.ticker] = trade
        return True

    def close_position(
        self,
        ticker: str,
        exit_date: datetime,
        exit_price: float,
        reason: str
    ):
        """Close an existing position"""
        if ticker not in self.positions:
            return

        trade = self.positions[ticker]
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate proceeds
        proceeds = exit_price * trade.quantity
        proceeds_after_cost = proceeds * (1 - self.transaction_cost)

        self.cash += proceeds_after_cost
        self.closed_trades.append(trade)
        del self.positions[ticker]

    def update_portfolio_value(self, current_date: datetime, prices: Dict[str, float]):
        """Update portfolio value based on current prices"""
        positions_value = sum(
            prices.get(ticker, trade.entry_price) * trade.quantity
            for ticker, trade in self.positions.items()
        )
        total_value = self.cash + positions_value
        self.portfolio_value_history.append((current_date, total_value))
        return total_value

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.closed_trades:
            return {}

        # Win rate
        winning_trades = [t for t in self.closed_trades if t.pnl() > 0]
        win_rate = len(winning_trades) / len(self.closed_trades)

        # Average win/loss
        avg_win = np.mean([t.pnl() for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.closed_trades if t.pnl() <= 0]
        avg_loss = np.mean([t.pnl() for t in losing_trades]) if losing_trades else 0

        # Total return
        total_pnl = sum(t.pnl() for t in self.closed_trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100

        # Sharpe ratio (from portfolio value history)
        if len(self.portfolio_value_history) > 1:
            values = [v for _, v in self.portfolio_value_history]
            returns = np.diff(values) / values[:-1]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        if len(self.portfolio_value_history) > 1:
            values = np.array([v for _, v in self.portfolio_value_history])
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100
        else:
            max_drawdown = 0

        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'final_portfolio_value': self.cash + sum(
                t.entry_price * t.quantity for t in self.positions.values()
            )
        }


def backtest_signals(
    predictions_df: pd.DataFrame,
    price_data: pd.DataFrame,
    config: BacktestConfig,
    horizon: str = '1month'
) -> Tuple[Portfolio, pd.DataFrame]:
    """
    Backtest trading signals

    Args:
        predictions_df: DataFrame with columns:
            - date, ticker, signal_{horizon}, confidence_{horizon}, expected_return_{horizon}
        price_data: DataFrame with columns:
            - date, ticker, close
        config: Backtest configuration
        horizon: Which horizon to backtest

    Returns:
        portfolio: Final portfolio state
        equity_curve: DataFrame with portfolio value over time
    """
    # Initialize portfolio
    portfolio = Portfolio(config.initial_capital, config.transaction_cost)

    # Get unique dates sorted
    dates = sorted(predictions_df['date'].unique())

    # Merge predictions with prices
    merged = predictions_df.merge(
        price_data[['date', 'ticker', 'close']],
        on=['date', 'ticker'],
        how='inner'
    )

    for current_date in dates:
        # Get predictions for this date
        day_predictions = merged[merged['date'] == current_date]

        # Get current prices
        current_prices = dict(zip(day_predictions['ticker'], day_predictions['close']))

        # Check existing positions for exit conditions
        for ticker in list(portfolio.positions.keys()):
            if ticker not in current_prices:
                continue

            trade = portfolio.positions[ticker]
            current_price = current_prices[ticker]

            # Calculate current return
            if trade.signal == 'BUY':
                current_return = (current_price - trade.entry_price) / trade.entry_price
            else:
                current_return = (trade.entry_price - current_price) / trade.entry_price

            # Check exit conditions
            should_exit = False
            exit_reason = None

            # Stop loss
            if current_return <= config.stop_loss:
                should_exit = True
                exit_reason = 'stop_loss'

            # Take profit
            elif current_return >= config.take_profit:
                should_exit = True
                exit_reason = 'take_profit'

            # Check if signal changed
            ticker_pred = day_predictions[day_predictions['ticker'] == ticker]
            if not ticker_pred.empty:
                new_signal = ticker_pred[f'signal_{horizon}'].iloc[0]
                if new_signal != trade.signal and new_signal != 'HOLD':
                    should_exit = True
                    exit_reason = 'signal_change'

            if should_exit:
                portfolio.close_position(ticker, current_date, current_price, exit_reason)

        # Look for new entry signals
        for _, row in day_predictions.iterrows():
            ticker = row['ticker']
            signal = row[f'signal_{horizon}']
            confidence = row.get(f'confidence_{horizon}', 0.5)
            price = row['close']

            # Skip if already in position or signal is HOLD
            if ticker in portfolio.positions or signal == 'HOLD':
                continue

            # Check confidence threshold
            if confidence < config.min_confidence:
                continue

            # Calculate position size
            position_value = portfolio.cash * config.position_size
            quantity = int(position_value / price)

            if quantity > 0:
                trade = Trade(
                    ticker=ticker,
                    entry_date=current_date,
                    entry_price=price,
                    signal=signal,
                    quantity=quantity,
                    confidence=confidence,
                    horizon=horizon
                )
                portfolio.open_position(trade)

        # Update portfolio value
        portfolio.update_portfolio_value(current_date, current_prices)

    # Close any remaining positions at last price
    if portfolio.positions and len(dates) > 0:
        last_date = dates[-1]
        last_prices = dict(zip(
            merged[merged['date'] == last_date]['ticker'],
            merged[merged['date'] == last_date]['close']
        ))

        for ticker in list(portfolio.positions.keys()):
            if ticker in last_prices:
                portfolio.close_position(
                    ticker, last_date, last_prices[ticker], 'backtest_end'
                )

    # Create equity curve DataFrame
    equity_curve = pd.DataFrame(
        portfolio.portfolio_value_history,
        columns=['date', 'portfolio_value']
    )

    return portfolio, equity_curve


def backtest_buy_and_hold(
    price_data: pd.DataFrame,
    initial_capital: float,
    tickers: Optional[List[str]] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Backtest buy-and-hold strategy for comparison

    Args:
        price_data: DataFrame with date, ticker, close
        initial_capital: Starting capital
        tickers: List of tickers to hold (equal weight)

    Returns:
        final_value: Final portfolio value
        equity_curve: Portfolio value over time
    """
    if tickers is None:
        tickers = price_data['ticker'].unique()

    # Get first and last dates
    first_date = price_data['date'].min()
    last_date = price_data['date'].max()

    # Get initial prices
    initial_prices = price_data[price_data['date'] == first_date]
    initial_prices = dict(zip(initial_prices['ticker'], initial_prices['close']))

    # Allocate capital equally across tickers
    capital_per_ticker = initial_capital / len(tickers)
    holdings = {}

    for ticker in tickers:
        if ticker in initial_prices:
            price = initial_prices[ticker]
            quantity = capital_per_ticker / price
            holdings[ticker] = quantity

    # Track portfolio value over time
    dates = sorted(price_data['date'].unique())
    equity_curve = []

    for date in dates:
        day_prices = price_data[price_data['date'] == date]
        current_prices = dict(zip(day_prices['ticker'], day_prices['close']))

        portfolio_value = sum(
            holdings.get(ticker, 0) * current_prices.get(ticker, 0)
            for ticker in tickers
        )

        equity_curve.append({'date': date, 'portfolio_value': portfolio_value})

    equity_curve_df = pd.DataFrame(equity_curve)
    final_value = equity_curve_df['portfolio_value'].iloc[-1]

    return final_value, equity_curve_df


def compare_strategies(
    signal_portfolio: Portfolio,
    signal_equity: pd.DataFrame,
    buyhold_equity: pd.DataFrame,
    config: BacktestConfig
) -> Dict:
    """Compare signal-based strategy vs buy-and-hold"""

    # Signal strategy metrics
    signal_metrics = signal_portfolio.get_performance_metrics()
    signal_final = signal_equity['portfolio_value'].iloc[-1]
    signal_return = ((signal_final - config.initial_capital) / config.initial_capital) * 100

    # Buy-and-hold metrics
    buyhold_final = buyhold_equity['portfolio_value'].iloc[-1]
    buyhold_return = ((buyhold_final - config.initial_capital) / config.initial_capital) * 100

    # Buy-and-hold Sharpe and drawdown
    values = buyhold_equity['portfolio_value'].values
    returns = np.diff(values) / values[:-1]
    buyhold_sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    buyhold_max_dd = np.min(drawdown) * 100

    comparison = {
        'signal_strategy': {
            'final_value': signal_final,
            'total_return_pct': signal_return,
            'sharpe_ratio': signal_metrics.get('sharpe_ratio', 0),
            'max_drawdown_pct': signal_metrics.get('max_drawdown_pct', 0),
            'win_rate': signal_metrics.get('win_rate', 0),
            'total_trades': signal_metrics.get('total_trades', 0)
        },
        'buy_and_hold': {
            'final_value': buyhold_final,
            'total_return_pct': buyhold_return,
            'sharpe_ratio': buyhold_sharpe,
            'max_drawdown_pct': buyhold_max_dd
        },
        'outperformance': signal_return - buyhold_return
    }

    return comparison


def plot_backtest_results(
    signal_equity: pd.DataFrame,
    buyhold_equity: pd.DataFrame,
    comparison: Dict,
    save_path: Optional[str] = None
):
    """Plot backtest results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Equity curves
    ax = axes[0, 0]
    ax.plot(signal_equity['date'], signal_equity['portfolio_value'],
            label='Signal Strategy', linewidth=2)
    ax.plot(buyhold_equity['date'], buyhold_equity['portfolio_value'],
            label='Buy & Hold', linewidth=2, alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Equity Curves')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Returns comparison
    ax = axes[0, 1]
    strategies = ['Signal Strategy', 'Buy & Hold']
    returns = [
        comparison['signal_strategy']['total_return_pct'],
        comparison['buy_and_hold']['total_return_pct']
    ]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax.bar(strategies, returns, color=colors, alpha=0.7)
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Total Returns Comparison')
    ax.grid(axis='y', alpha=0.3)

    # 3. Risk metrics
    ax = axes[1, 0]
    metrics = ['Sharpe Ratio', 'Max Drawdown (%)']
    signal_vals = [
        comparison['signal_strategy']['sharpe_ratio'],
        abs(comparison['signal_strategy']['max_drawdown_pct'])
    ]
    buyhold_vals = [
        comparison['buy_and_hold']['sharpe_ratio'],
        abs(comparison['buy_and_hold']['max_drawdown_pct'])
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, signal_vals, width, label='Signal Strategy', alpha=0.7)
    ax.bar(x + width/2, buyhold_vals, width, label='Buy & Hold', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Risk Metrics')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    BACKTEST SUMMARY
    {'='*40}

    Signal Strategy:
      Final Value:     ${comparison['signal_strategy']['final_value']:,.2f}
      Total Return:    {comparison['signal_strategy']['total_return_pct']:.2f}%
      Sharpe Ratio:    {comparison['signal_strategy']['sharpe_ratio']:.2f}
      Max Drawdown:    {comparison['signal_strategy']['max_drawdown_pct']:.2f}%
      Win Rate:        {comparison['signal_strategy']['win_rate']*100:.1f}%
      Total Trades:    {comparison['signal_strategy']['total_trades']}

    Buy & Hold:
      Final Value:     ${comparison['buy_and_hold']['final_value']:,.2f}
      Total Return:    {comparison['buy_and_hold']['total_return_pct']:.2f}%
      Sharpe Ratio:    {comparison['buy_and_hold']['sharpe_ratio']:.2f}
      Max Drawdown:    {comparison['buy_and_hold']['max_drawdown_pct']:.2f}%

    Outperformance:    {comparison['outperformance']:.2f}%
    """

    ax.text(0.1, 0.5, summary_text, family='monospace', fontsize=10,
            verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Backtest results saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    print("Backtesting module ready")
    print("Use backtest_signals() to simulate trading strategy")

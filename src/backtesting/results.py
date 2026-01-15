"""
Backtest results and analysis.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np

from .config import (
    BacktestConfig, PortfolioSnapshot, SimulatedTrade,
    CategoryPerformance, MonthlyPerformance
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    # Configuration
    config: BacktestConfig
    
    # Time range
    start_date: datetime
    end_date: datetime
    duration: timedelta
    
    # Portfolio performance
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    profit_factor: float
    expectancy: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: timedelta
    
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Time series
    equity_curve: List[PortfolioSnapshot]
    trades: List[SimulatedTrade]
    signals_generated: int
    signals_traded: int
    
    # Breakdown
    performance_by_category: Dict[str, CategoryPerformance]
    performance_by_month: Dict[str, MonthlyPerformance]
    
    def to_dict(self) -> dict:
        """Serialize results to dictionary."""
        return {
            "config": self.config.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration": str(self.duration),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration": str(self.max_drawdown_duration),
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "signals_generated": self.signals_generated,
            "signals_traded": self.signals_traded,
            "equity_curve": [s.to_dict() for s in self.equity_curve],
            "trades": [t.to_dict() for t in self.trades],
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        df = pd.DataFrame([s.to_dict() for s in self.equity_curve])
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        return df


class BacktestAnalyzer:
    """Analyzes backtest data to generate metrics and charts."""
    
    def __init__(
        self,
        config: BacktestConfig,
        equity_curve: List[PortfolioSnapshot],
        trades: List[SimulatedTrade],
        signals_generated: int,
        signals_traded: int
    ):
        self.config = config
        self.equity_curve = equity_curve
        self.trades = trades
        self.signals_generated = signals_generated
        self.signals_traded = signals_traded
    
    def analyze(self) -> BacktestResults:
        """Calculate all metrics and generate results object."""
        if not self.equity_curve:
            return self._empty_results()
        
        # Basic Capital metrics
        initial_capital = self.config.initial_capital
        final_capital = self.equity_curve[-1].total_value
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Trade Analysis
        closed_trades = [t for t in self.trades if t.realized_pnl is not None]
        winning_trades = [t for t in closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in closed_trades if t.realized_pnl <= 0]
        
        num_trades = len(closed_trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        win_rate = (num_wins / num_trades) if num_trades > 0 else 0.0
        
        avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0.0
        largest_win = max([t.realized_pnl for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t.realized_pnl for t in losing_trades]) if losing_trades else 0.0
        
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        expectancy = (avg_win * win_rate) + (avg_loss * (1 - win_rate))
        
        # Risk Metrics
        equity_series = pd.Series([s.total_value for s in self.equity_curve])
        returns = equity_series.pct_change().dropna()
        
        # Drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min())
        max_drawdown = abs((equity_series - rolling_max).min())
        
        # Calculate drawdown duration
        # (This is a simplified calculation)
        max_drawdown_duration = timedelta(0)
        if not drawdown.empty:
            is_dd = drawdown < 0
            # Identify streaks of True
            # For simplicity in this implementation, we'll skip exact duration calculation
            # without a more complex loop
            pass 
            
        # Sharpe Ratio (assuming daily steps for annualization factor of 252, or adjust by timestep)
        # We'll use the timestep from config to determine annualization factor
        seconds_per_year = 365 * 24 * 60 * 60
        steps_per_year = seconds_per_year / self.config.time_step.total_seconds()
        
        risk_free_rate = 0.04  # 4% annual
        rf_per_step = risk_free_rate / steps_per_year
        
        excess_returns = returns - rf_per_step
        std_dev = returns.std()
        
        if std_dev > 0:
            sharpe_ratio = np.sqrt(steps_per_year) * (excess_returns.mean() / std_dev)
        else:
            sharpe_ratio = 0.0
            
        # Sortino Ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std > 0:
            sortino_ratio = np.sqrt(steps_per_year) * (excess_returns.mean() / downside_std)
        else:
            sortino_ratio = 0.0
            
        # Calmar Ratio
        if max_drawdown_pct > 0:
            # Annualized return
            days = (self.config.end_date - self.config.start_date).days
            if days > 0:
                annual_return = (final_capital / initial_capital) ** (365 / days) - 1
                calmar_ratio = annual_return / max_drawdown_pct
            else:
                calmar_ratio = 0.0
        else:
            calmar_ratio = 0.0
            
        # Breakdowns (Placeholders for now)
        performance_by_category = {}
        performance_by_month = {}
        
        return BacktestResults(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration=self.config.end_date - self.config.start_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=num_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            equity_curve=self.equity_curve,
            trades=self.trades,
            signals_generated=self.signals_generated,
            signals_traded=self.signals_traded,
            performance_by_category=performance_by_category,
            performance_by_month=performance_by_month,
        )

    def _empty_results(self) -> BacktestResults:
        """Return empty results object."""
        return BacktestResults(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration=timedelta(0),
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
            total_return=0.0,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=timedelta(0),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            equity_curve=[],
            trades=[],
            signals_generated=0,
            signals_traded=0,
            performance_by_category={},
            performance_by_month={},
        )

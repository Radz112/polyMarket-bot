"""
Portfolio tracking and performance analytics.

Tracks portfolio value over time and calculates performance metrics.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import statistics

from src.database.postgres import DatabaseManager
from src.execution.paper.types import (
    PaperPosition,
    PaperTrade,
    PortfolioState,
    PortfolioSnapshot,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Tracks portfolio state and calculates performance metrics.
    
    Provides:
    - Current portfolio state snapshots
    - Historical portfolio values for equity curve
    - Trade history
    - Comprehensive performance metrics
    """
    
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        initial_balance: float = 10000.0
    ):
        """
        Initialize the portfolio tracker.
        
        Args:
            db: Database manager for persistence
            initial_balance: Initial portfolio balance for return calculations
        """
        self.db = db
        self.initial_balance = initial_balance
        
        # In-memory storage (for non-persistent mode)
        self._snapshots: List[PortfolioSnapshot] = []
        self._trades: List[PaperTrade] = []
    
    def record_snapshot(self, state: PortfolioState) -> None:
        """
        Record a portfolio snapshot for historical tracking.
        
        Args:
            state: Current portfolio state
        """
        snapshot = PortfolioSnapshot(
            id=None,
            timestamp=state.timestamp,
            cash_balance=state.cash_balance,
            positions_value=state.positions_value,
            total_value=state.total_value,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl=state.realized_pnl
        )
        self._snapshots.append(snapshot)
        logger.debug(f"Recorded snapshot: value=${state.total_value:.2f}")
    
    def record_trade(self, trade: PaperTrade) -> None:
        """
        Record a trade for history.
        
        Args:
            trade: The executed trade
        """
        self._trades.append(trade)
    
    def get_current_state(
        self,
        cash_balance: float,
        positions: List[PaperPosition],
        realized_pnl: float
    ) -> PortfolioState:
        """
        Get current portfolio state.
        
        Args:
            cash_balance: Current cash balance
            positions: List of open positions
            realized_pnl: Total realized P&L
            
        Returns:
            Current PortfolioState
        """
        positions_value = sum(p.market_value for p in positions)
        total_value = cash_balance + positions_value
        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_pnl = realized_pnl + unrealized_pnl
        return_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return PortfolioState(
            timestamp=datetime.utcnow(),
            cash_balance=cash_balance,
            positions=positions,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_pnl=total_pnl,
            return_pct=return_pct
        )
    
    def get_history(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        interval: str = "1h"
    ) -> List[PortfolioSnapshot]:
        """
        Get portfolio value history over time.
        
        Args:
            start: Start timestamp
            end: End timestamp (defaults to now)
            interval: Interval for grouping ("1h", "1d", etc.)
            
        Returns:
            List of portfolio snapshots for equity curve
        """
        if end is None:
            end = datetime.utcnow()
        
        # Filter snapshots by time range
        filtered = [
            s for s in self._snapshots
            if start <= s.timestamp <= end
        ]
        
        # Sort by timestamp
        filtered.sort(key=lambda x: x.timestamp)
        
        # Parse interval
        interval_seconds = self._parse_interval(interval)
        
        if len(filtered) <= 1 or interval_seconds == 0:
            return filtered
        
        # Downsample to interval
        result = []
        current_bucket = None
        
        for snap in filtered:
            bucket_start = self._get_bucket_start(snap.timestamp, interval_seconds)
            
            if current_bucket is None or bucket_start != current_bucket:
                result.append(snap)
                current_bucket = bucket_start
        
        return result
    
    def get_trade_history(
        self,
        limit: int = 100,
        market_id: Optional[str] = None
    ) -> List[PaperTrade]:
        """
        Get historical trades.
        
        Args:
            limit: Maximum trades to return
            market_id: Optional filter by market
            
        Returns:
            List of paper trades
        """
        trades = self._trades
        
        if market_id:
            trades = [t for t in trades if t.market_id == market_id]
        
        # Sort by timestamp descending (newest first)
        trades = sorted(trades, key=lambda x: x.timestamp, reverse=True)
        
        return trades[:limit]
    
    def get_performance_metrics(
        self,
        trades: Optional[List[PaperTrade]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: Optional list of trades (uses internal trades if None)
            
        Returns:
            PerformanceMetrics with win rate, Sharpe, drawdown, etc.
        """
        if trades is None:
            trades = self._trades
        
        # Filter to closed trades (those with realized P&L)
        closed_trades = [t for t in trades if t.realized_pnl is not None]
        
        if not closed_trades:
            return self._empty_metrics()
        
        # Calculate win/loss stats
        winning_trades = [t for t in closed_trades if t.realized_pnl > 0]
        losing_trades = [t for t in closed_trades if t.realized_pnl <= 0]
        
        total_trades = len(closed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = (
            sum(t.realized_pnl for t in winning_trades) / win_count
            if win_count > 0 else 0
        )
        avg_loss = (
            abs(sum(t.realized_pnl for t in losing_trades)) / loss_count
            if loss_count > 0 else 0
        )
        
        # Profit factor
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        total_return = sum(t.realized_pnl for t in closed_trades if t.realized_pnl)
        total_return_pct = (total_return / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Max drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown()
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe_ratio(closed_trades)
        
        # Average holding period
        holding_periods = [
            t.holding_period for t in closed_trades
            if t.holding_period is not None
        ]
        avg_holding = (
            sum(holding_periods, timedelta()) / len(holding_periods)
            if holding_periods else timedelta()
        )
        
        # Best/worst trades
        best_trade = max(closed_trades, key=lambda x: x.realized_pnl or 0)
        worst_trade = min(closed_trades, key=lambda x: x.realized_pnl or 0)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            avg_holding_period=avg_holding,
            best_trade=best_trade,
            worst_trade=worst_trade
        )
    
    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown from snapshots."""
        if len(self._snapshots) < 2:
            return 0.0, 0.0
        
        values = [s.total_value for s in sorted(self._snapshots, key=lambda x: x.timestamp)]
        
        peak = values[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
            
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct
        
        return max_dd, max_dd_pct
    
    def _calculate_sharpe_ratio(
        self,
        trades: List[PaperTrade],
        risk_free_rate: float = 0.05
    ) -> Optional[float]:
        """
        Calculate Sharpe ratio from trade returns.
        
        Args:
            trades: List of closed trades
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(trades) < 3:  # Need at least 3 trades for meaningful stats
            return None
        
        # Get returns as percentages
        returns = [
            (t.realized_pnl / (t.size * t.price)) * 100
            for t in trades
            if t.realized_pnl is not None and t.size * t.price > 0
        ]
        
        if len(returns) < 3:
            return None
        
        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return None
            
            # Annualize (assuming ~250 trading days)
            # This is a simplified calculation
            daily_rf = risk_free_rate / 250
            excess_return = mean_return - daily_rf
            
            sharpe = excess_return / std_return
            return sharpe
        except Exception:
            return None
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades exist."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            total_return=0.0,
            total_return_pct=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            avg_holding_period=timedelta(),
            best_trade=None,
            worst_trade=None
        )
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds."""
        if interval.endswith('s'):
            return int(interval[:-1])
        elif interval.endswith('m'):
            return int(interval[:-1]) * 60
        elif interval.endswith('h'):
            return int(interval[:-1]) * 3600
        elif interval.endswith('d'):
            return int(interval[:-1]) * 86400
        return 3600  # Default 1 hour
    
    def _get_bucket_start(self, timestamp: datetime, interval_seconds: int) -> datetime:
        """Get the start of the bucket containing this timestamp."""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (timestamp - epoch).total_seconds()
        bucket_start_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds
        return epoch + timedelta(seconds=bucket_start_seconds)
    
    def clear(self) -> None:
        """Clear all stored data."""
        self._snapshots.clear()
        self._trades.clear()

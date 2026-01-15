"""
Paper trading types and data structures.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any


class PositionStatus(str, Enum):
    """Paper position status."""
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"


@dataclass
class FillResult:
    """Result from simulating an order fill against an orderbook."""
    filled_size: float
    average_price: float
    slippage: float  # Difference from best price
    fees: float
    total_cost: float  # size * price + fees (for buys) or size * price - fees (for sells)
    unfilled_size: float  # Amount not filled (if limit hit)
    fill_time: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def was_partial(self) -> bool:
        """Check if this was a partial fill."""
        return self.unfilled_size > 0


@dataclass
class PaperPosition:
    """
    Represents an open paper trading position.
    
    Tracks size, entry price, and current value for P&L calculations.
    """
    id: str
    market_id: str
    market_name: str
    side: str  # "YES" or "NO"
    size: float
    entry_price: float
    current_price: float
    opened_at: datetime = field(default_factory=datetime.utcnow)
    
    # Optional metadata
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.size * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost of the position."""
        return self.size * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    @property
    def holding_period(self) -> timedelta:
        """Time since position was opened."""
        return datetime.utcnow() - self.opened_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "opened_at": self.opened_at.isoformat(),
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "holding_period_seconds": self.holding_period.total_seconds(),
            "signal_id": self.signal_id,
            "signal_score": self.signal_score,
        }


@dataclass
class PaperTrade:
    """
    Represents an executed paper trade.
    
    Records all details of the trade for historical analysis.
    """
    id: str
    market_id: str
    market_name: str
    side: str  # "YES" or "NO"
    action: str  # "BUY" or "SELL"
    size: float
    price: float
    fees: float
    total: float  # price * size +/- fees
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Link to signal that triggered the trade
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    
    # Position reference
    position_id: Optional[str] = None
    
    # Outcome (filled after position closes)
    realized_pnl: Optional[float] = None
    holding_period: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "side": self.side,
            "action": self.action,
            "size": self.size,
            "price": self.price,
            "fees": self.fees,
            "total": self.total,
            "timestamp": self.timestamp.isoformat(),
            "signal_id": self.signal_id,
            "signal_score": self.signal_score,
            "position_id": self.position_id,
            "realized_pnl": self.realized_pnl,
            "holding_period_seconds": self.holding_period.total_seconds() if self.holding_period else None,
        }


@dataclass
class PnLSummary:
    """Summary of profit and loss."""
    realized_pnl: float  # P&L from closed positions
    unrealized_pnl: float  # P&L from open positions
    total_pnl: float  # realized + unrealized
    total_fees: float  # Total fees paid
    return_pct: float  # Total return as percentage of initial balance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "return_pct": self.return_pct,
        }


@dataclass
class PortfolioState:
    """Current state of the paper trading portfolio."""
    timestamp: datetime
    cash_balance: float
    positions: List[PaperPosition]
    total_value: float  # cash + position values
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    return_pct: float  # Return on initial capital
    
    @property
    def positions_value(self) -> float:
        """Total value of all open positions."""
        return sum(p.market_value for p in self.positions)
    
    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash_balance": self.cash_balance,
            "positions": [p.to_dict() for p in self.positions],
            "positions_value": self.positions_value,
            "position_count": self.position_count,
            "total_value": self.total_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
        }


@dataclass
class PortfolioSnapshot:
    """Historical portfolio snapshot for equity curve tracking."""
    id: Optional[int]
    timestamp: datetime
    cash_balance: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for paper trading."""
    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Win/loss metrics
    win_rate: float  # winning_trades / total_trades
    average_win: float  # Average profit on winning trades
    average_loss: float  # Average loss on losing trades (absolute value)
    profit_factor: float  # gross_profit / gross_loss
    
    # Return metrics
    total_return: float  # Total profit/loss in dollars
    total_return_pct: float  # Total return as percentage
    
    # Risk metrics
    max_drawdown: float  # Maximum peak-to-trough decline
    max_drawdown_pct: float  # Max drawdown as percentage
    sharpe_ratio: Optional[float]  # Risk-adjusted return (None if insufficient data)
    
    # Timing
    avg_holding_period: timedelta
    
    # Best/worst trades
    best_trade: Optional[PaperTrade]
    worst_trade: Optional[PaperTrade]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_holding_period_seconds": self.avg_holding_period.total_seconds(),
            "best_trade": self.best_trade.to_dict() if self.best_trade else None,
            "worst_trade": self.worst_trade.to_dict() if self.worst_trade else None,
        }

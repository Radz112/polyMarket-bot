"""
Backtesting configuration and data models.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any


class SlippageModel(Enum):
    """Slippage simulation models."""
    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    ORDERBOOK = "orderbook"


class FeeModel(Enum):
    """Fee simulation models."""
    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"


class FillModel(Enum):
    """Order fill simulation models."""
    IMMEDIATE = "immediate"
    NEXT_BAR = "next_bar"
    REALISTIC = "realistic"


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENT = "percent"
    KELLY = "kelly"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Time range
    start_date: datetime
    end_date: datetime
    
    # Initial conditions
    initial_capital: float = 10000.0
    
    # Simulation settings
    time_step: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_bps: float = 10.0  # Basis points
    fee_model: FeeModel = FeeModel.PERCENTAGE
    fee_bps: float = 200.0  # 2% fees (Polymarket standard)
    
    # Execution settings
    fill_model: FillModel = FillModel.IMMEDIATE
    partial_fills: bool = False
    
    # Strategy settings
    signal_threshold: float = 70.0
    max_positions: int = 10
    position_size_method: PositionSizeMethod = PositionSizeMethod.FIXED
    position_size_value: float = 100.0  # $100 per position
    
    # Risk settings
    max_drawdown_halt: float = 0.20  # 20% drawdown halts backtest
    max_position_pct: float = 0.10  # Max 10% of portfolio per position
    
    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "time_step_seconds": self.time_step.total_seconds(),
            "slippage_model": self.slippage_model.value,
            "slippage_bps": self.slippage_bps,
            "fee_model": self.fee_model.value,
            "fee_bps": self.fee_bps,
            "fill_model": self.fill_model.value,
            "partial_fills": self.partial_fills,
            "signal_threshold": self.signal_threshold,
            "max_positions": self.max_positions,
            "position_size_method": self.position_size_method.value,
            "position_size_value": self.position_size_value,
            "max_drawdown_halt": self.max_drawdown_halt,
            "max_position_pct": self.max_position_pct,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BacktestConfig":
        """Create config from dictionary."""
        return cls(
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            initial_capital=data.get("initial_capital", 10000.0),
            time_step=timedelta(seconds=data.get("time_step_seconds", 60)),
            slippage_model=SlippageModel(data.get("slippage_model", "fixed")),
            slippage_bps=data.get("slippage_bps", 10.0),
            fee_model=FeeModel(data.get("fee_model", "percentage")),
            fee_bps=data.get("fee_bps", 200.0),
            fill_model=FillModel(data.get("fill_model", "immediate")),
            partial_fills=data.get("partial_fills", False),
            signal_threshold=data.get("signal_threshold", 70.0),
            max_positions=data.get("max_positions", 10),
            position_size_method=PositionSizeMethod(data.get("position_size_method", "fixed")),
            position_size_value=data.get("position_size_value", 100.0),
            max_drawdown_halt=data.get("max_drawdown_halt", 0.20),
            max_position_pct=data.get("max_position_pct", 0.10),
        )


@dataclass
class PriceSnapshot:
    """Price data at a point in time."""
    market_id: str
    timestamp: datetime
    yes_price: float
    no_price: float
    yes_volume: float = 0.0
    no_volume: float = 0.0


@dataclass
class OrderbookSnapshot:
    """Orderbook state at a point in time."""
    market_id: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{price, size}, ...]
    asks: List[Dict[str, float]]  # [{price, size}, ...]


@dataclass
class SimulatedPosition:
    """A position in the simulated portfolio."""
    market_id: str
    side: str  # "YES" or "NO"
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return (self.current_price - self.entry_price) * self.size
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class SimulatedTrade:
    """A trade in the simulated portfolio."""
    id: str
    market_id: str
    market_name: str
    side: str  # "YES" or "NO"
    action: str  # "BUY", "SELL", "RESOLVE"
    size: float
    price: float
    fees: float
    timestamp: datetime
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    realized_pnl: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Serialize trade."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "side": self.side,
            "action": self.action,
            "size": self.size,
            "price": self.price,
            "fees": self.fees,
            "timestamp": self.timestamp.isoformat(),
            "signal_id": self.signal_id,
            "signal_score": self.signal_score,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    num_positions: int
    drawdown: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize snapshot."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "num_positions": self.num_positions,
            "drawdown": self.drawdown,
        }


@dataclass
class MarketResolution:
    """Market resolution data."""
    market_id: str
    resolution_time: datetime
    outcome: str  # "YES" or "NO"


@dataclass 
class CategoryPerformance:
    """Performance breakdown by category."""
    category: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_trade: float


@dataclass
class MonthlyPerformance:
    """Monthly performance breakdown."""
    month: str  # YYYY-MM format
    trades: int
    pnl: float
    return_pct: float
    win_rate: float

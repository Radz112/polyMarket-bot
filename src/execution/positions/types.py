"""
Position tracking types and data structures.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any


class PositionStatus(str, Enum):
    """Position lifecycle status."""
    OPEN = "open"
    CLOSING = "closing"  # Marked for close but not yet executed
    CLOSED = "closed"
    RESOLVED = "resolved"  # Closed by market resolution


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"


class AlertType(str, Enum):
    """Position alert types."""
    PNL_GAIN = "pnl_gain"
    PNL_LOSS = "pnl_loss"
    TIME_HELD = "time"
    RESOLUTION_NEAR = "resolution"
    CORRELATION_CONFLICT = "correlation"


@dataclass
class Position:
    """
    Comprehensive position tracking with real-time data.
    
    Enhanced from PaperPosition with bid/ask, category, and correlations.
    """
    id: str
    market_id: str
    market_name: str
    category: str
    side: str  # "YES" or "NO"
    
    # Core data
    size: float
    entry_price: float
    entry_time: datetime
    
    # Current state (updated in real-time)
    current_price: float
    current_bid: float = 0.0
    current_ask: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    # Risk metrics
    distance_to_resolution: Optional[timedelta] = None
    
    # Linked data
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    correlated_positions: List[str] = field(default_factory=list)
    
    # Status
    status: PositionStatus = PositionStatus.OPEN
    
    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return self.size * self.entry_price
    
    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.size * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in dollars."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    @property
    def time_held(self) -> timedelta:
        """Time since position opened."""
        return datetime.utcnow() - self.entry_time
    
    @property
    def spread(self) -> float:
        """Current bid-ask spread."""
        if self.current_ask > 0 and self.current_bid > 0:
            return self.current_ask - self.current_bid
        return 0.0
    
    def update_price(self, price: float, bid: float = None, ask: float = None) -> None:
        """Update current price and recalculate metrics."""
        self.current_price = price
        if bid is not None:
            self.current_bid = bid
        if ask is not None:
            self.current_ask = ask
        self.last_update = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/API."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "category": self.category,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "last_update": self.last_update.isoformat(),
            "cost_basis": self.cost_basis,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "time_held_seconds": self.time_held.total_seconds(),
            "distance_to_resolution_seconds": (
                self.distance_to_resolution.total_seconds() 
                if self.distance_to_resolution else None
            ),
            "signal_id": self.signal_id,
            "signal_score": self.signal_score,
            "correlated_positions": self.correlated_positions,
            "status": self.status.value,
        }


@dataclass
class ClosedPosition:
    """A position that has been closed."""
    position: Position
    exit_price: float
    exit_time: datetime
    realized_pnl: float
    holding_period: timedelta
    exit_reason: str  # "signal", "stop_loss", "take_profit", "manual", "resolution"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "position": self.position.to_dict(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat(),
            "realized_pnl": self.realized_pnl,
            "holding_period_seconds": self.holding_period.total_seconds(),
            "exit_reason": self.exit_reason,
        }


@dataclass
class PositionAlert:
    """Alert generated for a position."""
    alert_type: AlertType
    severity: AlertSeverity
    position_id: str
    market_id: str
    message: str
    recommended_action: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Context data
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/notification."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "position_id": self.position_id,
            "market_id": self.market_id,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat(),
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
        }


@dataclass
class ConcentrationReport:
    """Portfolio concentration analysis."""
    largest_position: Optional[Position]
    largest_position_pct: float
    largest_category: str
    largest_category_pct: float
    correlation_clusters: List[List[str]]  # Groups of correlated position IDs
    diversification_score: float  # 0-100 (higher = more diversified)
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize report."""
        return {
            "largest_position": self.largest_position.to_dict() if self.largest_position else None,
            "largest_position_pct": self.largest_position_pct,
            "largest_category": self.largest_category,
            "largest_category_pct": self.largest_category_pct,
            "correlation_clusters": self.correlation_clusters,
            "diversification_score": self.diversification_score,
            "warnings": self.warnings,
        }


@dataclass
class PositionSnapshot:
    """Point-in-time position state for history tracking."""
    id: Optional[int]
    position_id: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class PositionAnalytics:
    """Analytics for a single position."""
    position_id: str
    entry_timing_score: float  # 0-100 (was entry well-timed?)
    max_gain: float  # Maximum unrealized gain while held
    max_gain_time: Optional[datetime]
    max_loss: float  # Maximum unrealized loss while held
    max_loss_time: Optional[datetime]
    current_vs_optimal: float  # Current P&L vs optimal exit P&L
    price_volatility: float  # Standard deviation of price during hold
    snapshots_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize analytics."""
        return {
            "position_id": self.position_id,
            "entry_timing_score": self.entry_timing_score,
            "max_gain": self.max_gain,
            "max_gain_time": self.max_gain_time.isoformat() if self.max_gain_time else None,
            "max_loss": self.max_loss,
            "max_loss_time": self.max_loss_time.isoformat() if self.max_loss_time else None,
            "current_vs_optimal": self.current_vs_optimal,
            "price_volatility": self.price_volatility,
            "snapshots_count": self.snapshots_count,
        }


@dataclass
class ExposureSummary:
    """Summary of portfolio exposure."""
    total_exposure: float
    net_exposure_by_underlying: Dict[str, float]
    exposure_by_category: Dict[str, float]
    largest_single_exposure: float
    largest_category_exposure: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary."""
        return {
            "total_exposure": self.total_exposure,
            "net_exposure_by_underlying": self.net_exposure_by_underlying,
            "exposure_by_category": self.exposure_by_category,
            "largest_single_exposure": self.largest_single_exposure,
            "largest_category_exposure": self.largest_category_exposure,
        }

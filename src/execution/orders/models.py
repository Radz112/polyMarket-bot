"""
Order management models and types.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"      # Created, not yet submitted
    SUBMITTED = "submitted"  # Sent to exchange
    PARTIAL = "partial"      # Partially filled
    FILLED = "filled"        # Fully filled
    CANCELLED = "cancelled"  # Cancelled by user
    REJECTED = "rejected"    # Rejected by exchange/validator
    EXPIRED = "expired"      # GTT order expired


@dataclass
class Fill:
    """Individual fill for an order."""
    price: float
    size: float
    timestamp: datetime
    fee: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp.isoformat(),
            "fee": self.fee
        }


@dataclass
class Order:
    """
    Order for execution.
    
    Tracks full order lifecycle from creation to fill.
    """
    id: str
    market_id: str
    market_name: str
    side: str           # "YES" or "NO"
    action: str         # "BUY" or "SELL"
    order_type: OrderType
    
    # Sizing
    size: float
    filled_size: float = 0.0
    
    # Pricing
    limit_price: Optional[float] = None
    average_fill_price: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Fees
    estimated_fees: float = 0.0
    actual_fees: float = 0.0
    
    # Linking
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    position_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For split orders
    
    # Execution details
    fills: List[Fill] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    
    @property
    def remaining_size(self) -> float:
        """Size remaining to fill."""
        return self.size - self.filled_size
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in (
            OrderStatus.FILLED, OrderStatus.CANCELLED, 
            OrderStatus.REJECTED, OrderStatus.EXPIRED
        )
    
    @property
    def notional_value(self) -> float:
        """Estimated order value."""
        price = self.limit_price or self.average_fill_price or 0.5
        return self.size * price
    
    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.size == 0:
            return 0.0
        return (self.filled_size / self.size) * 100
    
    def add_fill(self, fill: Fill) -> None:
        """Add a fill to this order."""
        self.fills.append(fill)
        self.filled_size += fill.size
        self.actual_fees += fill.fee
        
        # Update average fill price
        if self.fills:
            total_value = sum(f.price * f.size for f in self.fills)
            total_size = sum(f.size for f in self.fills)
            if total_size > 0:
                self.average_fill_price = total_value / total_size
        
        # Update status
        if self.filled_size >= self.size:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        elif self.filled_size > 0:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self, reason: str = None) -> None:
        """Cancel this order."""
        self.status = OrderStatus.CANCELLED
        if reason:
            self.rejection_reason = reason
    
    def reject(self, reason: str) -> None:
        """Reject this order."""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
    
    def submit(self) -> None:
        """Mark order as submitted."""
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/API."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "market_name": self.market_name,
            "side": self.side,
            "action": self.action,
            "order_type": self.order_type.value,
            "size": self.size,
            "filled_size": self.filled_size,
            "remaining_size": self.remaining_size,
            "limit_price": self.limit_price,
            "average_fill_price": self.average_fill_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "estimated_fees": self.estimated_fees,
            "actual_fees": self.actual_fees,
            "signal_id": self.signal_id,
            "position_id": self.position_id,
            "parent_order_id": self.parent_order_id,
            "fills": [f.to_dict() for f in self.fills],
            "rejection_reason": self.rejection_reason,
            "fill_rate": self.fill_rate,
        }


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order: Order
    message: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "order": self.order.to_dict(),
            "message": self.message,
            "error": self.error
        }


@dataclass
class ValidationResult:
    """Result of order validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


@dataclass
class ExecutionPlan:
    """Plan for order execution."""
    original_order: Order
    child_orders: List[Order] = field(default_factory=list)
    strategy: str = "single"  # "single", "split", "twap"
    estimated_fill_price: float = 0.0
    estimated_slippage: float = 0.0
    estimated_time_to_fill: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    
    @property
    def order_count(self) -> int:
        """Number of orders in plan."""
        return len(self.child_orders) if self.child_orders else 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_order_id": self.original_order.id,
            "child_orders": [o.id for o in self.child_orders],
            "strategy": self.strategy,
            "estimated_fill_price": self.estimated_fill_price,
            "estimated_slippage": self.estimated_slippage,
            "estimated_time_to_fill_seconds": self.estimated_time_to_fill.total_seconds(),
            "order_count": self.order_count
        }

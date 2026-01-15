"""
Divergence types and data structures for arbitrage detection.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import uuid

from src.models import Orderbook, Signal, SignalType


class DivergenceType(Enum):
    """Types of divergences that can be detected between correlated markets."""
    PRICE_SPREAD = "price_spread"              # Equivalent markets differ in price
    THRESHOLD_VIOLATION = "threshold_violation" # A > B when A should be <= B
    INVERSE_SUM = "inverse_sum"                # YES + NO != 100%
    LAGGING_MARKET = "lagging_market"          # One moved, other didn't
    CORRELATION_BREAK = "correlation_break"    # Historical correlation broke
    LEAD_LAG_OPPORTUNITY = "lead_lag"          # Leader moved, follower hasn't


class DivergenceStatus(str, Enum):
    """Lifecycle status of a divergence."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TRADED = "traded"
    FALSE_POSITIVE = "false_positive"


@dataclass
class Divergence:
    """
    Represents a detected divergence between correlated markets.

    A divergence indicates a potential trading opportunity where market prices
    have deviated from their expected relationship.
    """
    id: str
    divergence_type: DivergenceType
    detected_at: datetime

    # Markets involved
    market_ids: List[str]
    correlation_id: Optional[str] = None
    rule_id: Optional[str] = None

    # Current state
    current_prices: Dict[str, float] = field(default_factory=dict)
    current_orderbooks: Dict[str, Orderbook] = field(default_factory=dict)

    # Divergence metrics
    expected_relationship: str = ""  # "A ≈ B" or "A <= B" or "A + B ≈ 1"
    expected_value: float = 0.0
    actual_value: float = 0.0
    divergence_amount: float = 0.0   # Absolute difference
    divergence_pct: float = 0.0      # Percentage difference

    # Opportunity assessment
    direction: str = ""              # "BUY_A", "SELL_B", "BUY_A_SELL_B"
    is_arbitrage: bool = False       # True if guaranteed profit
    profit_potential: float = 0.0    # Expected profit in dollars per $1 risked
    max_executable_size: float = 0.0 # Liquidity-limited size

    # Confidence
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)

    # Lifecycle
    status: DivergenceStatus = DivergenceStatus.ACTIVE
    expires_at: Optional[datetime] = None

    # Metadata for additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default expiration if not provided."""
        if self.expires_at is None:
            # Default: expire in 5 minutes
            self.expires_at = self.detected_at + timedelta(minutes=5)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique divergence ID."""
        return str(uuid.uuid4())

    def is_expired(self) -> bool:
        """Check if this divergence has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def mark_traded(self) -> None:
        """Mark this divergence as having been traded."""
        self.status = DivergenceStatus.TRADED

    def mark_expired(self) -> None:
        """Mark this divergence as expired."""
        self.status = DivergenceStatus.EXPIRED

    def mark_false_positive(self, reason: str = "") -> None:
        """Mark this divergence as a false positive."""
        self.status = DivergenceStatus.FALSE_POSITIVE
        if reason:
            self.supporting_evidence.append(f"False positive: {reason}")

    def to_signal(self) -> Signal:
        """
        Convert this divergence to a tradeable Signal.

        Maps divergence types to signal types and packages the relevant
        information for the trading system.
        """
        # Map divergence type to signal type
        signal_type_map = {
            DivergenceType.PRICE_SPREAD: SignalType.DIVERGENCE,
            DivergenceType.THRESHOLD_VIOLATION: SignalType.THRESHOLD_VIOLATION,
            DivergenceType.INVERSE_SUM: SignalType.INVERSE_SUM_DEVIATION,
            DivergenceType.LAGGING_MARKET: SignalType.LAGGING_MARKET,
            DivergenceType.CORRELATION_BREAK: SignalType.DIVERGENCE,
            DivergenceType.LEAD_LAG_OPPORTUNITY: SignalType.LAGGING_MARKET,
        }

        signal_type = signal_type_map.get(self.divergence_type, SignalType.DIVERGENCE)

        # Calculate score based on profitability and confidence
        # Higher is better
        score = self.profit_potential * self.confidence
        if self.is_arbitrage:
            score *= 2.0  # Boost guaranteed profits

        return Signal(
            id=self.id,
            timestamp=self.detected_at,
            signal_type=signal_type,
            market_ids=self.market_ids,
            divergence_amount=self.divergence_amount,
            expected_value=self.expected_value,
            actual_value=self.actual_value,
            confidence=self.confidence,
            score=score,
            metadata={
                "divergence_type": self.divergence_type.value,
                "direction": self.direction,
                "is_arbitrage": self.is_arbitrage,
                "profit_potential": self.profit_potential,
                "max_executable_size": self.max_executable_size,
                "expected_relationship": self.expected_relationship,
                "supporting_evidence": self.supporting_evidence,
                "correlation_id": self.correlation_id,
                "rule_id": self.rule_id,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "divergence_type": self.divergence_type.value,
            "detected_at": self.detected_at.isoformat(),
            "market_ids": self.market_ids,
            "correlation_id": self.correlation_id,
            "rule_id": self.rule_id,
            "current_prices": self.current_prices,
            "expected_relationship": self.expected_relationship,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "divergence_amount": self.divergence_amount,
            "divergence_pct": self.divergence_pct,
            "direction": self.direction,
            "is_arbitrage": self.is_arbitrage,
            "profit_potential": self.profit_potential,
            "max_executable_size": self.max_executable_size,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "status": self.status.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DivergenceConfig:
    """Configuration for divergence detection thresholds."""
    # Minimum spread to flag as divergence (e.g., 0.02 = 2 cents)
    min_divergence_threshold: float = 0.02

    # Minimum liquidity required (in dollars)
    min_liquidity_threshold: float = 100.0

    # Minimum correlation confidence to consider
    min_correlation_confidence: float = 0.8

    # Lookback window for lagging market detection (seconds)
    lagging_lookback_seconds: int = 300

    # Window for correlation break detection (minutes)
    correlation_break_window_minutes: int = 60

    # Debounce settings
    debounce_seconds: float = 1.0  # Min time between same divergence alerts

    # Expiration settings
    default_expiry_minutes: int = 5

    # Per-type thresholds (override defaults)
    type_thresholds: Dict[DivergenceType, float] = field(default_factory=dict)

    def get_threshold(self, divergence_type: DivergenceType) -> float:
        """Get the threshold for a specific divergence type."""
        return self.type_thresholds.get(divergence_type, self.min_divergence_threshold)

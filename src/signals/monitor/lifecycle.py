"""
Signal lifecycle state machine.

Manages signal states from detection through resolution.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

from src.signals.scoring.types import ScoredSignal

logger = logging.getLogger(__name__)


class SignalState(str, Enum):
    """Possible states for a signal."""
    DETECTED = "detected"          # Just found
    ACTIVE = "active"              # Being monitored
    TRADED = "traded"              # We acted on it
    CONVERGED = "converged"        # Gap closed naturally
    EXPIRED = "expired"            # Timed out
    INVALIDATED = "invalidated"    # Correlation broke or data issue


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: datetime
    from_state: SignalState
    to_state: SignalState
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalLifecycle:
    """
    Manages the lifecycle of a signal from detection to resolution.

    Tracks state transitions and provides methods to check
    expiration and convergence conditions.
    """

    # Default expiration times by divergence type
    DEFAULT_EXPIRY_SECONDS = {
        "threshold_violation": 60,      # 1 minute - arbitrage
        "inverse_sum": 120,             # 2 minutes
        "price_spread": 300,            # 5 minutes
        "lead_lag_opportunity": 120,    # 2 minutes
        "lagging_market": 600,          # 10 minutes
        "correlation_break": 1800,      # 30 minutes
    }

    # Convergence thresholds (percentage of original divergence)
    CONVERGENCE_THRESHOLD = 0.20  # 80% closed = converged

    def __init__(
        self,
        signal: ScoredSignal,
        expiry_seconds: Optional[int] = None,
    ):
        self.signal = signal
        self.signal_id = signal.divergence.id
        self.state = SignalState.DETECTED
        self.history: List[StateTransition] = []

        # Set expiry time
        dtype = signal.divergence.divergence_type.value
        default_expiry = self.DEFAULT_EXPIRY_SECONDS.get(dtype, 300)
        self.expiry_seconds = expiry_seconds or default_expiry

        # Track original divergence for convergence detection
        self.original_divergence_pct = signal.divergence.divergence_pct
        self.original_divergence_amount = signal.divergence.divergence_amount

        # Timestamps
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(seconds=self.expiry_seconds)
        self.last_updated = self.created_at

        # Record initial state
        self._record_transition(
            SignalState.DETECTED,
            SignalState.DETECTED,
            "Signal created"
        )

    def transition(
        self,
        new_state: SignalState,
        reason: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Transition to a new state.

        Returns True if transition was valid, False otherwise.
        """
        # Validate transition
        if not self._is_valid_transition(self.state, new_state):
            logger.warning(
                f"Invalid state transition: {self.state} -> {new_state} "
                f"for signal {self.signal_id}"
            )
            return False

        old_state = self.state
        self.state = new_state
        self.last_updated = datetime.utcnow()

        self._record_transition(old_state, new_state, reason, metadata)

        logger.info(
            f"Signal {self.signal_id}: {old_state.value} -> {new_state.value} ({reason})"
        )

        return True

    def _record_transition(
        self,
        from_state: SignalState,
        to_state: SignalState,
        reason: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a state transition in history."""
        self.history.append(StateTransition(
            timestamp=datetime.utcnow(),
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata or {},
        ))

    def _is_valid_transition(
        self,
        from_state: SignalState,
        to_state: SignalState
    ) -> bool:
        """Check if a state transition is valid."""
        # Define valid transitions
        valid_transitions = {
            SignalState.DETECTED: {
                SignalState.ACTIVE,
                SignalState.EXPIRED,
                SignalState.INVALIDATED,
            },
            SignalState.ACTIVE: {
                SignalState.TRADED,
                SignalState.CONVERGED,
                SignalState.EXPIRED,
                SignalState.INVALIDATED,
            },
            SignalState.TRADED: {
                SignalState.CONVERGED,
                SignalState.EXPIRED,
            },
            # Terminal states - no transitions out
            SignalState.CONVERGED: set(),
            SignalState.EXPIRED: set(),
            SignalState.INVALIDATED: set(),
        }

        return to_state in valid_transitions.get(from_state, set())

    def activate(self, reason: str = "Passed filters") -> bool:
        """Activate the signal for monitoring."""
        return self.transition(SignalState.ACTIVE, reason)

    def mark_traded(
        self,
        trade_id: str = None,
        trade_size: float = 0.0
    ) -> bool:
        """Mark signal as traded."""
        return self.transition(
            SignalState.TRADED,
            "Trade executed",
            metadata={"trade_id": trade_id, "trade_size": trade_size}
        )

    def mark_converged(self, final_divergence: float) -> bool:
        """Mark signal as converged."""
        return self.transition(
            SignalState.CONVERGED,
            f"Divergence converged to {final_divergence:.4f}",
            metadata={"final_divergence": final_divergence}
        )

    def mark_expired(self) -> bool:
        """Mark signal as expired."""
        return self.transition(SignalState.EXPIRED, "Signal timed out")

    def mark_invalidated(self, reason: str) -> bool:
        """Mark signal as invalidated."""
        return self.transition(SignalState.INVALIDATED, reason)

    def should_expire(self) -> bool:
        """Check if signal should be expired based on time."""
        return datetime.utcnow() >= self.expires_at

    def has_converged(
        self,
        current_divergence_pct: float,
        threshold: float = None
    ) -> bool:
        """
        Check if divergence has closed enough to be considered converged.

        Args:
            current_divergence_pct: Current divergence percentage
            threshold: Convergence threshold (default: 80% closed)

        Returns:
            True if converged
        """
        if threshold is None:
            threshold = self.CONVERGENCE_THRESHOLD

        if self.original_divergence_pct <= 0:
            return current_divergence_pct <= 0

        # Calculate how much of the gap has closed
        remaining = current_divergence_pct / self.original_divergence_pct

        # Converged if remaining is below threshold
        return remaining <= threshold

    def is_terminal(self) -> bool:
        """Check if signal is in a terminal state."""
        return self.state in {
            SignalState.CONVERGED,
            SignalState.EXPIRED,
            SignalState.INVALIDATED,
        }

    def is_active(self) -> bool:
        """Check if signal is actively being monitored."""
        return self.state in {SignalState.DETECTED, SignalState.ACTIVE, SignalState.TRADED}

    def time_remaining(self) -> timedelta:
        """Get time remaining before expiration."""
        remaining = self.expires_at - datetime.utcnow()
        return remaining if remaining.total_seconds() > 0 else timedelta(seconds=0)

    def extend_expiry(self, additional_seconds: int) -> None:
        """Extend the expiration time."""
        self.expires_at = self.expires_at + timedelta(seconds=additional_seconds)
        logger.debug(f"Extended expiry for {self.signal_id} by {additional_seconds}s")

    def update_signal(self, new_signal: ScoredSignal) -> None:
        """Update the underlying signal with new data."""
        self.signal = new_signal
        self.last_updated = datetime.utcnow()

    def get_age_seconds(self) -> float:
        """Get signal age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    def get_state_duration(self) -> float:
        """Get time spent in current state in seconds."""
        if not self.history:
            return 0.0

        last_transition = self.history[-1]
        return (datetime.utcnow() - last_transition.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lifecycle for storage/transmission."""
        return {
            "signal_id": self.signal_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "original_divergence_pct": self.original_divergence_pct,
            "original_divergence_amount": self.original_divergence_amount,
            "age_seconds": self.get_age_seconds(),
            "time_remaining_seconds": self.time_remaining().total_seconds(),
            "history": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "reason": t.reason,
                }
                for t in self.history
            ],
            "signal": self.signal.to_dict() if hasattr(self.signal, 'to_dict') else {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], signal: ScoredSignal) -> "SignalLifecycle":
        """Recreate lifecycle from serialized data."""
        lifecycle = cls(signal)
        lifecycle.state = SignalState(data["state"])
        lifecycle.created_at = datetime.fromisoformat(data["created_at"])
        lifecycle.expires_at = datetime.fromisoformat(data["expires_at"])
        lifecycle.last_updated = datetime.fromisoformat(data["last_updated"])
        lifecycle.original_divergence_pct = data["original_divergence_pct"]
        lifecycle.original_divergence_amount = data["original_divergence_amount"]

        # Reconstruct history
        lifecycle.history = [
            StateTransition(
                timestamp=datetime.fromisoformat(h["timestamp"]),
                from_state=SignalState(h["from_state"]),
                to_state=SignalState(h["to_state"]),
                reason=h["reason"],
            )
            for h in data.get("history", [])
        ]

        return lifecycle


class LifecycleManager:
    """
    Manages multiple signal lifecycles.

    Provides bulk operations and queries across all active signals.
    """

    def __init__(self):
        self.lifecycles: Dict[str, SignalLifecycle] = {}

    def add(self, signal: ScoredSignal) -> SignalLifecycle:
        """Add a new signal and create its lifecycle."""
        lifecycle = SignalLifecycle(signal)
        self.lifecycles[signal.divergence.id] = lifecycle
        return lifecycle

    def get(self, signal_id: str) -> Optional[SignalLifecycle]:
        """Get lifecycle by signal ID."""
        return self.lifecycles.get(signal_id)

    def remove(self, signal_id: str) -> Optional[SignalLifecycle]:
        """Remove and return a lifecycle."""
        return self.lifecycles.pop(signal_id, None)

    def get_active(self) -> List[SignalLifecycle]:
        """Get all non-terminal lifecycles."""
        return [lc for lc in self.lifecycles.values() if lc.is_active()]

    def get_by_state(self, state: SignalState) -> List[SignalLifecycle]:
        """Get all lifecycles in a specific state."""
        return [lc for lc in self.lifecycles.values() if lc.state == state]

    def get_expired(self) -> List[SignalLifecycle]:
        """Get all lifecycles that should be expired."""
        return [lc for lc in self.lifecycles.values()
                if lc.should_expire() and lc.is_active()]

    def cleanup_terminal(self) -> List[SignalLifecycle]:
        """Remove all terminal lifecycles and return them."""
        terminal = [lc for lc in self.lifecycles.values() if lc.is_terminal()]
        for lc in terminal:
            del self.lifecycles[lc.signal_id]
        return terminal

    def count_by_state(self) -> Dict[str, int]:
        """Get count of signals by state."""
        counts = {}
        for state in SignalState:
            counts[state.value] = len(self.get_by_state(state))
        return counts

    def __len__(self) -> int:
        return len(self.lifecycles)

    def __contains__(self, signal_id: str) -> bool:
        return signal_id in self.lifecycles

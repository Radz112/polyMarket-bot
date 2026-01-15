"""
Signal blacklist and whitelist management.

Allows blocking specific markets/correlations and boosting priority signals.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List, Any, FrozenSet
from enum import Enum

from src.signals.scoring.types import ScoredSignal

logger = logging.getLogger(__name__)


class BlockReason(str, Enum):
    """Reason for blocking a signal."""
    MANUAL = "manual"
    POOR_PERFORMANCE = "poor_performance"
    DATA_QUALITY = "data_quality"
    MARKET_SUSPENDED = "market_suspended"
    CORRELATION_INVALID = "correlation_invalid"
    TEMPORARY = "temporary"


class BoostReason(str, Enum):
    """Reason for boosting a signal."""
    MANUAL = "manual"
    HIGH_PRIORITY = "high_priority"
    STRONG_PERFORMANCE = "strong_performance"
    USER_PREFERENCE = "user_preference"


@dataclass
class BlockEntry:
    """Entry in the blacklist."""
    reason: BlockReason
    added_at: datetime
    expires_at: Optional[datetime] = None
    note: str = ""

    def is_expired(self) -> bool:
        """Check if this block has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class BoostEntry:
    """Entry in the whitelist."""
    reason: BoostReason
    score_boost: float
    added_at: datetime
    expires_at: Optional[datetime] = None
    note: str = ""

    def is_expired(self) -> bool:
        """Check if this boost has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class SignalBlacklist:
    """
    Manages blocked markets and correlations.

    Supports:
    - Blocking individual markets
    - Blocking specific market pairs/correlations
    - Temporary blocks with expiration
    - Reason tracking for audit
    """

    def __init__(self):
        # Blocked individual markets
        self._blocked_markets: Dict[str, BlockEntry] = {}

        # Blocked market pairs (stored as frozenset)
        self._blocked_pairs: Dict[FrozenSet[str], BlockEntry] = {}

        # Blocked correlation IDs
        self._blocked_correlations: Dict[str, BlockEntry] = {}

        # Statistics
        self._block_count = 0
        self._signals_blocked = 0

    def block_market(
        self,
        market_id: str,
        reason: BlockReason,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Block a specific market."""
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._blocked_markets[market_id] = BlockEntry(
            reason=reason,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._block_count += 1
        logger.info(f"Blocked market {market_id}: {reason.value}")

    def block_pair(
        self,
        market_a: str,
        market_b: str,
        reason: BlockReason,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Block a specific market pair."""
        pair = frozenset([market_a, market_b])
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._blocked_pairs[pair] = BlockEntry(
            reason=reason,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._block_count += 1
        logger.info(f"Blocked pair {market_a}-{market_b}: {reason.value}")

    def block_correlation(
        self,
        correlation_id: str,
        reason: BlockReason,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Block a specific correlation by ID."""
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._blocked_correlations[correlation_id] = BlockEntry(
            reason=reason,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._block_count += 1
        logger.info(f"Blocked correlation {correlation_id}: {reason.value}")

    def unblock_market(self, market_id: str) -> bool:
        """Unblock a market. Returns True if it was blocked."""
        if market_id in self._blocked_markets:
            del self._blocked_markets[market_id]
            logger.info(f"Unblocked market {market_id}")
            return True
        return False

    def unblock_pair(self, market_a: str, market_b: str) -> bool:
        """Unblock a market pair."""
        pair = frozenset([market_a, market_b])
        if pair in self._blocked_pairs:
            del self._blocked_pairs[pair]
            logger.info(f"Unblocked pair {market_a}-{market_b}")
            return True
        return False

    def unblock_correlation(self, correlation_id: str) -> bool:
        """Unblock a correlation."""
        if correlation_id in self._blocked_correlations:
            del self._blocked_correlations[correlation_id]
            logger.info(f"Unblocked correlation {correlation_id}")
            return True
        return False

    def is_blocked(self, signal: ScoredSignal) -> tuple[bool, Optional[str]]:
        """
        Check if a signal is blocked.

        Returns (is_blocked, reason_string).
        """
        # Clean expired entries
        self._cleanup_expired()

        market_ids = signal.divergence.market_ids

        # Check individual markets
        for market_id in market_ids:
            if market_id in self._blocked_markets:
                entry = self._blocked_markets[market_id]
                self._signals_blocked += 1
                return True, f"Market {market_id} blocked: {entry.reason.value}"

        # Check pairs
        if len(market_ids) >= 2:
            pair = frozenset(market_ids)
            if pair in self._blocked_pairs:
                entry = self._blocked_pairs[pair]
                self._signals_blocked += 1
                return True, f"Market pair blocked: {entry.reason.value}"

        # Check correlation ID if available
        if hasattr(signal.divergence, 'correlation_id') and signal.divergence.correlation_id:
            if signal.divergence.correlation_id in self._blocked_correlations:
                entry = self._blocked_correlations[signal.divergence.correlation_id]
                self._signals_blocked += 1
                return True, f"Correlation blocked: {entry.reason.value}"

        return False, None

    def filter(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """Filter out blocked signals."""
        return [s for s in signals if not self.is_blocked(s)[0]]

    def _cleanup_expired(self) -> None:
        """Remove expired block entries."""
        self._blocked_markets = {
            k: v for k, v in self._blocked_markets.items()
            if not v.is_expired()
        }
        self._blocked_pairs = {
            k: v for k, v in self._blocked_pairs.items()
            if not v.is_expired()
        }
        self._blocked_correlations = {
            k: v for k, v in self._blocked_correlations.items()
            if not v.is_expired()
        }

    def get_blocked_markets(self) -> List[Dict[str, Any]]:
        """Get list of blocked markets with details."""
        self._cleanup_expired()
        return [
            {
                "market_id": market_id,
                "reason": entry.reason.value,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "note": entry.note,
            }
            for market_id, entry in self._blocked_markets.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get blacklist statistics."""
        self._cleanup_expired()
        return {
            "blocked_markets": len(self._blocked_markets),
            "blocked_pairs": len(self._blocked_pairs),
            "blocked_correlations": len(self._blocked_correlations),
            "total_blocks_added": self._block_count,
            "signals_blocked": self._signals_blocked,
        }


class SignalWhitelist:
    """
    Manages boosted/prioritized markets and correlations.

    Whitelisted signals get priority processing and score boosts.
    """

    def __init__(self, default_boost: float = 10.0):
        self.default_boost = default_boost

        # Boosted individual markets
        self._boosted_markets: Dict[str, BoostEntry] = {}

        # Boosted market pairs
        self._boosted_pairs: Dict[FrozenSet[str], BoostEntry] = {}

        # Boosted correlation IDs
        self._boosted_correlations: Dict[str, BoostEntry] = {}

        # Statistics
        self._boost_count = 0
        self._signals_boosted = 0

    def boost_market(
        self,
        market_id: str,
        reason: BoostReason,
        score_boost: Optional[float] = None,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Add boost for a specific market."""
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._boosted_markets[market_id] = BoostEntry(
            reason=reason,
            score_boost=score_boost or self.default_boost,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._boost_count += 1
        logger.info(f"Boosted market {market_id}: {reason.value}")

    def boost_pair(
        self,
        market_a: str,
        market_b: str,
        reason: BoostReason,
        score_boost: Optional[float] = None,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Add boost for a specific market pair."""
        pair = frozenset([market_a, market_b])
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._boosted_pairs[pair] = BoostEntry(
            reason=reason,
            score_boost=score_boost or self.default_boost,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._boost_count += 1
        logger.info(f"Boosted pair {market_a}-{market_b}: {reason.value}")

    def boost_correlation(
        self,
        correlation_id: str,
        reason: BoostReason,
        score_boost: Optional[float] = None,
        duration_hours: Optional[float] = None,
        note: str = ""
    ) -> None:
        """Add boost for a specific correlation."""
        expires_at = None
        if duration_hours is not None:
            expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        self._boosted_correlations[correlation_id] = BoostEntry(
            reason=reason,
            score_boost=score_boost or self.default_boost,
            added_at=datetime.utcnow(),
            expires_at=expires_at,
            note=note,
        )
        self._boost_count += 1
        logger.info(f"Boosted correlation {correlation_id}: {reason.value}")

    def remove_boost_market(self, market_id: str) -> bool:
        """Remove boost from a market."""
        if market_id in self._boosted_markets:
            del self._boosted_markets[market_id]
            return True
        return False

    def remove_boost_pair(self, market_a: str, market_b: str) -> bool:
        """Remove boost from a market pair."""
        pair = frozenset([market_a, market_b])
        if pair in self._boosted_pairs:
            del self._boosted_pairs[pair]
            return True
        return False

    def get_boost(self, signal: ScoredSignal) -> tuple[float, Optional[str]]:
        """
        Get boost amount for a signal.

        Returns (boost_amount, reason_string).
        """
        self._cleanup_expired()

        total_boost = 0.0
        reasons = []

        market_ids = signal.divergence.market_ids

        # Check individual markets
        for market_id in market_ids:
            if market_id in self._boosted_markets:
                entry = self._boosted_markets[market_id]
                total_boost += entry.score_boost
                reasons.append(f"Market {market_id}: +{entry.score_boost}")

        # Check pairs
        if len(market_ids) >= 2:
            pair = frozenset(market_ids)
            if pair in self._boosted_pairs:
                entry = self._boosted_pairs[pair]
                total_boost += entry.score_boost
                reasons.append(f"Pair: +{entry.score_boost}")

        # Check correlation ID
        if hasattr(signal.divergence, 'correlation_id') and signal.divergence.correlation_id:
            if signal.divergence.correlation_id in self._boosted_correlations:
                entry = self._boosted_correlations[signal.divergence.correlation_id]
                total_boost += entry.score_boost
                reasons.append(f"Correlation: +{entry.score_boost}")

        if total_boost > 0:
            self._signals_boosted += 1

        return total_boost, "; ".join(reasons) if reasons else None

    def is_whitelisted(self, signal: ScoredSignal) -> bool:
        """Check if signal has any whitelist boost."""
        boost, _ = self.get_boost(signal)
        return boost > 0

    def _cleanup_expired(self) -> None:
        """Remove expired boost entries."""
        self._boosted_markets = {
            k: v for k, v in self._boosted_markets.items()
            if not v.is_expired()
        }
        self._boosted_pairs = {
            k: v for k, v in self._boosted_pairs.items()
            if not v.is_expired()
        }
        self._boosted_correlations = {
            k: v for k, v in self._boosted_correlations.items()
            if not v.is_expired()
        }

    def get_boosted_markets(self) -> List[Dict[str, Any]]:
        """Get list of boosted markets with details."""
        self._cleanup_expired()
        return [
            {
                "market_id": market_id,
                "reason": entry.reason.value,
                "score_boost": entry.score_boost,
                "added_at": entry.added_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "note": entry.note,
            }
            for market_id, entry in self._boosted_markets.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get whitelist statistics."""
        self._cleanup_expired()
        return {
            "boosted_markets": len(self._boosted_markets),
            "boosted_pairs": len(self._boosted_pairs),
            "boosted_correlations": len(self._boosted_correlations),
            "total_boosts_added": self._boost_count,
            "signals_boosted": self._signals_boosted,
        }

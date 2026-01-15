"""
Signal deduplication logic.

Prevents duplicate signals from being processed.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Set

from src.signals.scoring.types import ScoredSignal

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Configuration for signal deduplication."""
    # Same markets = duplicate
    require_same_markets: bool = True

    # Same divergence type = duplicate
    require_same_type: bool = True

    # Divergence amount must be within this percentage to be duplicate
    amount_similarity_threshold: float = 0.20  # 20%

    # Time window for considering duplicates
    time_window_seconds: int = 300  # 5 minutes

    # Minimum time between updates for same signal
    update_cooldown_seconds: int = 5


class SignalDeduplicator:
    """
    Detects and handles duplicate signals.

    A signal is considered a duplicate if:
    - Same markets involved
    - Same divergence type
    - Similar divergence amount (within threshold)
    - Detected within the time window
    """

    def __init__(self, config: DeduplicationConfig = None):
        self.config = config or DeduplicationConfig()

        # Track recently seen signals for fast lookup
        # Key: (frozenset of market_ids, divergence_type)
        # Value: (signal_id, timestamp, divergence_amount)
        self._recent_signals: Dict[
            Tuple[frozenset, str],
            Tuple[str, datetime, float]
        ] = {}

        # Track signal fingerprints for exact match detection
        self._fingerprints: Dict[str, str] = {}  # fingerprint -> signal_id

    def is_duplicate(
        self,
        new_signal: ScoredSignal,
        active_signals: Dict[str, ScoredSignal]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if signal is a duplicate of an existing active signal.

        Args:
            new_signal: The newly detected signal
            active_signals: Dictionary of currently active signals

        Returns:
            Tuple of (is_duplicate, existing_signal_id or None)
        """
        # Generate fingerprint for the new signal
        fingerprint = self._generate_fingerprint(new_signal)

        # Check exact fingerprint match first (fast path)
        if fingerprint in self._fingerprints:
            existing_id = self._fingerprints[fingerprint]
            if existing_id in active_signals:
                logger.debug(f"Exact duplicate found: {existing_id}")
                return True, existing_id

        # Check against active signals
        for signal_id, existing in active_signals.items():
            if self._is_similar(new_signal, existing):
                logger.debug(f"Similar signal found: {signal_id}")
                return True, signal_id

        # Check recent signals cache
        cache_key = self._get_cache_key(new_signal)
        if cache_key in self._recent_signals:
            cached_id, cached_time, cached_amount = self._recent_signals[cache_key]

            # Check time window
            age = (datetime.utcnow() - cached_time).total_seconds()
            if age < self.config.time_window_seconds:
                # Check amount similarity
                if self._amounts_similar(
                    new_signal.divergence.divergence_amount,
                    cached_amount
                ):
                    if cached_id in active_signals:
                        return True, cached_id

        return False, None

    def register_signal(self, signal: ScoredSignal) -> None:
        """
        Register a new signal for future deduplication.

        Call this after a signal has been accepted.
        """
        signal_id = signal.divergence.id

        # Add fingerprint
        fingerprint = self._generate_fingerprint(signal)
        self._fingerprints[fingerprint] = signal_id

        # Add to recent signals cache
        cache_key = self._get_cache_key(signal)
        self._recent_signals[cache_key] = (
            signal_id,
            datetime.utcnow(),
            signal.divergence.divergence_amount,
        )

        logger.debug(f"Registered signal {signal_id} for deduplication")

    def unregister_signal(self, signal: ScoredSignal) -> None:
        """
        Remove a signal from deduplication tracking.

        Call this when a signal is closed or expired.
        """
        signal_id = signal.divergence.id

        # Remove fingerprint
        fingerprint = self._generate_fingerprint(signal)
        if fingerprint in self._fingerprints:
            del self._fingerprints[fingerprint]

        # Remove from cache
        cache_key = self._get_cache_key(signal)
        if cache_key in self._recent_signals:
            if self._recent_signals[cache_key][0] == signal_id:
                del self._recent_signals[cache_key]

    def should_update(
        self,
        signal_id: str,
        last_update: datetime
    ) -> bool:
        """
        Check if enough time has passed to update a signal.

        Prevents excessive updates for the same signal.
        """
        age = (datetime.utcnow() - last_update).total_seconds()
        return age >= self.config.update_cooldown_seconds

    def merge_signals(
        self,
        existing: ScoredSignal,
        new: ScoredSignal
    ) -> ScoredSignal:
        """
        Merge a new signal into an existing one.

        Strategy:
        - Keep higher overall score
        - Update prices to current
        - Extend expiry if new signal is fresher
        - Combine supporting evidence

        Returns the merged signal.
        """
        # Start with whichever has higher score
        if new.overall_score > existing.overall_score:
            merged = new
            other = existing
        else:
            merged = existing
            other = new

        # Update prices from newer signal
        merged.divergence.current_prices = new.divergence.current_prices.copy()
        merged.divergence.current_orderbooks = new.divergence.current_orderbooks

        # Update divergence metrics from newer
        merged.divergence.divergence_pct = new.divergence.divergence_pct
        merged.divergence.divergence_amount = new.divergence.divergence_amount
        merged.divergence.profit_potential = new.divergence.profit_potential
        merged.divergence.max_executable_size = new.divergence.max_executable_size

        # Combine supporting evidence (deduplicated)
        existing_evidence = set(existing.divergence.supporting_evidence or [])
        new_evidence = set(new.divergence.supporting_evidence or [])
        merged.divergence.supporting_evidence = list(existing_evidence | new_evidence)

        # Update timestamp
        merged.scored_at = datetime.utcnow()

        logger.debug(
            f"Merged signal: score {existing.overall_score:.1f} + "
            f"{new.overall_score:.1f} -> {merged.overall_score:.1f}"
        )

        return merged

    def _generate_fingerprint(self, signal: ScoredSignal) -> str:
        """Generate a unique fingerprint for a signal."""
        div = signal.divergence
        markets = ",".join(sorted(div.market_ids))
        dtype = div.divergence_type.value

        # Round divergence to reduce noise
        div_rounded = round(div.divergence_pct, 3)

        return f"{markets}:{dtype}:{div_rounded}"

    def _get_cache_key(
        self,
        signal: ScoredSignal
    ) -> Tuple[frozenset, str]:
        """Get cache key for a signal."""
        markets = frozenset(signal.divergence.market_ids)
        dtype = signal.divergence.divergence_type.value
        return (markets, dtype)

    def _is_similar(
        self,
        new: ScoredSignal,
        existing: ScoredSignal
    ) -> bool:
        """Check if two signals are similar enough to be duplicates."""
        # Check markets
        if self.config.require_same_markets:
            if set(new.divergence.market_ids) != set(existing.divergence.market_ids):
                return False

        # Check type
        if self.config.require_same_type:
            if new.divergence.divergence_type != existing.divergence.divergence_type:
                return False

        # Check amount similarity
        if not self._amounts_similar(
            new.divergence.divergence_amount,
            existing.divergence.divergence_amount
        ):
            return False

        return True

    def _amounts_similar(
        self,
        amount1: float,
        amount2: float
    ) -> bool:
        """Check if two divergence amounts are similar."""
        if amount1 == 0 and amount2 == 0:
            return True

        if amount1 == 0 or amount2 == 0:
            return False

        # Calculate relative difference
        larger = max(abs(amount1), abs(amount2))
        diff = abs(amount1 - amount2)
        relative_diff = diff / larger

        return relative_diff <= self.config.amount_similarity_threshold

    def cleanup_old_entries(self, max_age_seconds: int = 3600) -> int:
        """
        Remove old entries from caches.

        Returns count of removed entries.
        """
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        removed = 0

        # Clean recent signals cache
        to_remove = []
        for key, (signal_id, timestamp, _) in self._recent_signals.items():
            if timestamp < cutoff:
                to_remove.append(key)

        for key in to_remove:
            del self._recent_signals[key]
            removed += 1

        if removed > 0:
            logger.debug(f"Cleaned up {removed} old deduplication entries")

        return removed

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            "fingerprints_tracked": len(self._fingerprints),
            "recent_signals_cached": len(self._recent_signals),
        }


class MarketSetDeduplicator:
    """
    Fast deduplicator optimized for market set matching.

    Uses bloom filter-like approach for O(1) duplicate detection
    when dealing with many signals.
    """

    def __init__(self, time_window_seconds: int = 300):
        self.time_window_seconds = time_window_seconds

        # Market set to signal mapping
        self._market_sets: Dict[frozenset, Set[str]] = {}

        # Signal timestamps
        self._timestamps: Dict[str, datetime] = {}

    def check_and_register(
        self,
        signal_id: str,
        market_ids: list[str]
    ) -> Optional[str]:
        """
        Check for duplicate and register if new.

        Returns existing signal_id if duplicate, None if new.
        """
        market_set = frozenset(market_ids)
        now = datetime.utcnow()

        # Clean old entries for this market set
        self._clean_market_set(market_set)

        # Check for existing signal
        if market_set in self._market_sets:
            existing_ids = self._market_sets[market_set]
            if existing_ids:
                # Return first (oldest) active signal
                return next(iter(existing_ids))

        # Register new signal
        if market_set not in self._market_sets:
            self._market_sets[market_set] = set()

        self._market_sets[market_set].add(signal_id)
        self._timestamps[signal_id] = now

        return None

    def _clean_market_set(self, market_set: frozenset) -> None:
        """Remove expired signals from a market set."""
        if market_set not in self._market_sets:
            return

        cutoff = datetime.utcnow() - timedelta(seconds=self.time_window_seconds)
        expired = set()

        for signal_id in self._market_sets[market_set]:
            if signal_id in self._timestamps:
                if self._timestamps[signal_id] < cutoff:
                    expired.add(signal_id)

        for signal_id in expired:
            self._market_sets[market_set].discard(signal_id)
            self._timestamps.pop(signal_id, None)

    def remove(self, signal_id: str, market_ids: list[str]) -> None:
        """Remove a signal from tracking."""
        market_set = frozenset(market_ids)
        if market_set in self._market_sets:
            self._market_sets[market_set].discard(signal_id)
        self._timestamps.pop(signal_id, None)

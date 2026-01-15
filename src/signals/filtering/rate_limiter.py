"""
Signal rate limiting to prevent alert fatigue.

Implements various rate limiting strategies for signal emission.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Deque, Set, FrozenSet
from collections import deque, defaultdict
from enum import Enum

from src.signals.scoring.types import ScoredSignal

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategy."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Global limits
    global_max_per_minute: int = 20
    global_max_per_hour: int = 100

    # Per-market limits
    per_market_max_per_minute: int = 5
    per_market_max_per_hour: int = 20

    # Per-pair limits
    per_pair_max_per_minute: int = 3
    per_pair_max_per_hour: int = 10

    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Token bucket config (if using token bucket)
    bucket_capacity: int = 10
    refill_rate_per_second: float = 0.5

    # Burst allowance
    allow_burst: bool = True
    burst_size: int = 5
    burst_cooldown_seconds: float = 60.0

    # Critical signal bypass
    critical_bypass_enabled: bool = True
    critical_score_threshold: float = 90.0


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    reason: str
    wait_seconds: float = 0.0
    current_count: int = 0
    limit: int = 0


class SlidingWindowCounter:
    """Sliding window rate counter."""

    def __init__(self, window_seconds: float, max_count: int):
        self.window_seconds = window_seconds
        self.max_count = max_count
        self._events: Deque[datetime] = deque()

    def record(self) -> None:
        """Record an event."""
        self._cleanup()
        self._events.append(datetime.utcnow())

    def can_proceed(self) -> tuple[bool, int]:
        """Check if we can proceed. Returns (can_proceed, current_count)."""
        self._cleanup()
        current = len(self._events)
        return current < self.max_count, current

    def wait_time(self) -> float:
        """Get time to wait until we can proceed (0 if can proceed now)."""
        self._cleanup()
        if len(self._events) < self.max_count:
            return 0.0

        # Time until oldest event expires
        oldest = self._events[0]
        expiry = oldest + timedelta(seconds=self.window_seconds)
        wait = (expiry - datetime.utcnow()).total_seconds()
        return max(0.0, wait)

    def _cleanup(self) -> None:
        """Remove expired events."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float  # tokens per second
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = datetime.utcnow()

    def try_consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available."""
        self._refill()
        if self._tokens >= tokens:
            return 0.0
        needed = tokens - self._tokens
        return needed / self.refill_rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = datetime.utcnow()
        elapsed = (now - self._last_refill).total_seconds()
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now


class SignalRateLimiter:
    """
    Rate limits signal emission to prevent alert fatigue.

    Supports multiple limiting strategies and granularities.
    """

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()

        # Global rate limiters
        self._global_minute = SlidingWindowCounter(60, config.global_max_per_minute if config else 20)
        self._global_hour = SlidingWindowCounter(3600, config.global_max_per_hour if config else 100)

        # Per-market rate limiters
        self._market_minute: Dict[str, SlidingWindowCounter] = {}
        self._market_hour: Dict[str, SlidingWindowCounter] = {}

        # Per-pair rate limiters
        self._pair_minute: Dict[FrozenSet[str], SlidingWindowCounter] = {}
        self._pair_hour: Dict[FrozenSet[str], SlidingWindowCounter] = {}

        # Token bucket (alternative strategy)
        self._token_bucket = TokenBucket(
            config.bucket_capacity if config else 10,
            config.refill_rate_per_second if config else 0.5
        )

        # Burst tracking
        self._burst_events: Deque[datetime] = deque()
        self._in_burst_cooldown = False
        self._burst_cooldown_until: Optional[datetime] = None

        # Statistics
        self._total_checked = 0
        self._total_allowed = 0
        self._total_rejected = 0
        self._rejections_by_reason: Dict[str, int] = defaultdict(int)

    def check(self, signal: ScoredSignal) -> RateLimitResult:
        """
        Check if a signal can be emitted.

        Returns RateLimitResult with decision.
        """
        self._total_checked += 1

        # Critical signal bypass
        if self.config.critical_bypass_enabled:
            if signal.overall_score >= self.config.critical_score_threshold:
                self._total_allowed += 1
                return RateLimitResult(
                    allowed=True,
                    reason="Critical signal bypass",
                    current_count=0,
                    limit=0,
                )

        # Check burst cooldown
        if self._in_burst_cooldown:
            if datetime.utcnow() < self._burst_cooldown_until:
                self._total_rejected += 1
                self._rejections_by_reason["burst_cooldown"] += 1
                wait = (self._burst_cooldown_until - datetime.utcnow()).total_seconds()
                return RateLimitResult(
                    allowed=False,
                    reason="In burst cooldown period",
                    wait_seconds=wait,
                )
            else:
                self._in_burst_cooldown = False

        # Use appropriate strategy
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(signal)
        else:
            return self._check_sliding_window(signal)

    def _check_sliding_window(self, signal: ScoredSignal) -> RateLimitResult:
        """Check using sliding window strategy."""
        # Check global limits first
        can_global_minute, count_minute = self._global_minute.can_proceed()
        if not can_global_minute:
            self._total_rejected += 1
            self._rejections_by_reason["global_minute"] += 1
            return RateLimitResult(
                allowed=False,
                reason="Global per-minute limit reached",
                wait_seconds=self._global_minute.wait_time(),
                current_count=count_minute,
                limit=self.config.global_max_per_minute,
            )

        can_global_hour, count_hour = self._global_hour.can_proceed()
        if not can_global_hour:
            self._total_rejected += 1
            self._rejections_by_reason["global_hour"] += 1
            return RateLimitResult(
                allowed=False,
                reason="Global per-hour limit reached",
                wait_seconds=self._global_hour.wait_time(),
                current_count=count_hour,
                limit=self.config.global_max_per_hour,
            )

        # Check per-market limits
        for market_id in signal.divergence.market_ids:
            result = self._check_market_limit(market_id)
            if not result.allowed:
                self._total_rejected += 1
                return result

        # Check per-pair limits
        pair = frozenset(signal.divergence.market_ids)
        result = self._check_pair_limit(pair)
        if not result.allowed:
            self._total_rejected += 1
            return result

        self._total_allowed += 1
        return RateLimitResult(
            allowed=True,
            reason="Within all rate limits",
        )

    def _check_market_limit(self, market_id: str) -> RateLimitResult:
        """Check per-market rate limits."""
        # Initialize counters if needed
        if market_id not in self._market_minute:
            self._market_minute[market_id] = SlidingWindowCounter(
                60, self.config.per_market_max_per_minute
            )
            self._market_hour[market_id] = SlidingWindowCounter(
                3600, self.config.per_market_max_per_hour
            )

        can_minute, count = self._market_minute[market_id].can_proceed()
        if not can_minute:
            self._rejections_by_reason["market_minute"] += 1
            return RateLimitResult(
                allowed=False,
                reason=f"Per-market minute limit for {market_id}",
                wait_seconds=self._market_minute[market_id].wait_time(),
                current_count=count,
                limit=self.config.per_market_max_per_minute,
            )

        can_hour, count = self._market_hour[market_id].can_proceed()
        if not can_hour:
            self._rejections_by_reason["market_hour"] += 1
            return RateLimitResult(
                allowed=False,
                reason=f"Per-market hour limit for {market_id}",
                wait_seconds=self._market_hour[market_id].wait_time(),
                current_count=count,
                limit=self.config.per_market_max_per_hour,
            )

        return RateLimitResult(allowed=True, reason="OK")

    def _check_pair_limit(self, pair: FrozenSet[str]) -> RateLimitResult:
        """Check per-pair rate limits."""
        if len(pair) < 2:
            return RateLimitResult(allowed=True, reason="Single market")

        # Initialize counters if needed
        if pair not in self._pair_minute:
            self._pair_minute[pair] = SlidingWindowCounter(
                60, self.config.per_pair_max_per_minute
            )
            self._pair_hour[pair] = SlidingWindowCounter(
                3600, self.config.per_pair_max_per_hour
            )

        can_minute, count = self._pair_minute[pair].can_proceed()
        if not can_minute:
            self._rejections_by_reason["pair_minute"] += 1
            return RateLimitResult(
                allowed=False,
                reason=f"Per-pair minute limit",
                wait_seconds=self._pair_minute[pair].wait_time(),
                current_count=count,
                limit=self.config.per_pair_max_per_minute,
            )

        can_hour, count = self._pair_hour[pair].can_proceed()
        if not can_hour:
            self._rejections_by_reason["pair_hour"] += 1
            return RateLimitResult(
                allowed=False,
                reason=f"Per-pair hour limit",
                wait_seconds=self._pair_hour[pair].wait_time(),
                current_count=count,
                limit=self.config.per_pair_max_per_hour,
            )

        return RateLimitResult(allowed=True, reason="OK")

    def _check_token_bucket(self, signal: ScoredSignal) -> RateLimitResult:
        """Check using token bucket strategy."""
        if self._token_bucket.try_consume(1):
            self._total_allowed += 1
            return RateLimitResult(
                allowed=True,
                reason="Token consumed",
            )
        else:
            self._total_rejected += 1
            self._rejections_by_reason["token_bucket"] += 1
            return RateLimitResult(
                allowed=False,
                reason="No tokens available",
                wait_seconds=self._token_bucket.wait_time(1),
            )

    def record(self, signal: ScoredSignal) -> None:
        """
        Record that a signal was emitted.

        Call this after signal emission to update counters.
        """
        # Record in global counters
        self._global_minute.record()
        self._global_hour.record()

        # Record in per-market counters
        for market_id in signal.divergence.market_ids:
            if market_id not in self._market_minute:
                self._market_minute[market_id] = SlidingWindowCounter(
                    60, self.config.per_market_max_per_minute
                )
                self._market_hour[market_id] = SlidingWindowCounter(
                    3600, self.config.per_market_max_per_hour
                )
            self._market_minute[market_id].record()
            self._market_hour[market_id].record()

        # Record in per-pair counters
        pair = frozenset(signal.divergence.market_ids)
        if len(pair) >= 2:
            if pair not in self._pair_minute:
                self._pair_minute[pair] = SlidingWindowCounter(
                    60, self.config.per_pair_max_per_minute
                )
                self._pair_hour[pair] = SlidingWindowCounter(
                    3600, self.config.per_pair_max_per_hour
                )
            self._pair_minute[pair].record()
            self._pair_hour[pair].record()

        # Track burst
        self._track_burst()

    def _track_burst(self) -> None:
        """Track burst activity."""
        if not self.config.allow_burst:
            return

        now = datetime.utcnow()
        self._burst_events.append(now)

        # Keep only recent burst events
        cutoff = now - timedelta(seconds=5)  # 5-second burst window
        while self._burst_events and self._burst_events[0] < cutoff:
            self._burst_events.popleft()

        # Check if we're in a burst
        if len(self._burst_events) >= self.config.burst_size:
            self._in_burst_cooldown = True
            self._burst_cooldown_until = now + timedelta(
                seconds=self.config.burst_cooldown_seconds
            )
            self._burst_events.clear()
            logger.warning(
                f"Burst detected, entering cooldown until {self._burst_cooldown_until}"
            )

    def filter(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """
        Filter signals by rate limit.

        Returns signals that pass rate limiting.
        Records allowed signals automatically.
        """
        allowed = []
        for signal in signals:
            result = self.check(signal)
            if result.allowed:
                allowed.append(signal)
                self.record(signal)
        return allowed

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "total_checked": self._total_checked,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                self._total_rejected / self._total_checked
                if self._total_checked > 0 else 0
            ),
            "rejections_by_reason": dict(self._rejections_by_reason),
            "in_burst_cooldown": self._in_burst_cooldown,
            "tracked_markets": len(self._market_minute),
            "tracked_pairs": len(self._pair_minute),
        }

    def reset(self) -> None:
        """Reset all rate limiters."""
        self._global_minute = SlidingWindowCounter(60, self.config.global_max_per_minute)
        self._global_hour = SlidingWindowCounter(3600, self.config.global_max_per_hour)
        self._market_minute.clear()
        self._market_hour.clear()
        self._pair_minute.clear()
        self._pair_hour.clear()
        self._burst_events.clear()
        self._in_burst_cooldown = False
        self._burst_cooldown_until = None
        self._total_checked = 0
        self._total_allowed = 0
        self._total_rejected = 0
        self._rejections_by_reason.clear()

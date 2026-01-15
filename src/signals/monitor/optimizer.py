"""
Monitor performance optimization utilities.

Provides indexing and batching for efficient divergence detection.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Iterator

from src.signals.scoring.types import ScoredSignal
from src.models import MarketCorrelation

logger = logging.getLogger(__name__)


@dataclass
class MarketPriority:
    """Priority information for a market."""
    market_id: str
    priority_score: float
    last_update: datetime
    active_signal_count: int
    recent_price_move: float = 0.0
    has_arbitrage: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for the monitor optimizer."""
    # Batching
    default_batch_size: int = 50
    max_batch_size: int = 100

    # Priority scoring
    arbitrage_priority_boost: float = 10.0
    high_score_threshold: float = 70.0
    high_score_priority_boost: float = 5.0
    recent_move_threshold: float = 0.02  # 2%
    recent_move_priority_boost: float = 3.0

    # Update frequency
    hot_market_update_interval: float = 1.0  # seconds
    normal_market_update_interval: float = 5.0


class MonitorOptimizer:
    """
    Optimizes monitoring performance through indexing and prioritization.

    Features:
    - Market -> Correlation index for fast lookups
    - Market prioritization based on signal importance
    - Batch processing for parallel divergence checks
    """

    def __init__(self, config: OptimizerConfig = None):
        self.config = config or OptimizerConfig()

        # Index: market_id -> list of correlation_ids involving this market
        self.market_to_correlations: Dict[str, List[str]] = defaultdict(list)

        # Index: correlation_id -> MarketCorrelation
        self.correlations: Dict[str, MarketCorrelation] = {}

        # Hot markets needing frequent updates
        self.hot_markets: Set[str] = set()

        # Market priority scores
        self.market_priorities: Dict[str, MarketPriority] = {}

        # Recent price moves for spike detection
        self.recent_moves: Dict[str, List[tuple[datetime, float]]] = defaultdict(list)

        # Statistics
        self._index_build_time: Optional[datetime] = None
        self._correlation_count = 0

    def build_market_index(
        self,
        correlations: List[MarketCorrelation]
    ) -> None:
        """
        Build index: market_id -> correlation_ids.

        Allows fast lookup of which correlations to check
        when a specific market's price changes.
        """
        start_time = datetime.utcnow()

        # Clear existing index
        self.market_to_correlations.clear()
        self.correlations.clear()

        for corr in correlations:
            corr_id = corr.id
            self.correlations[corr_id] = corr

            # Index both markets
            self.market_to_correlations[corr.market_a_id].append(corr_id)
            self.market_to_correlations[corr.market_b_id].append(corr_id)

        self._index_build_time = datetime.utcnow()
        self._correlation_count = len(correlations)

        build_duration = (self._index_build_time - start_time).total_seconds() * 1000
        logger.info(
            f"Built market index: {len(correlations)} correlations, "
            f"{len(self.market_to_correlations)} markets indexed in {build_duration:.1f}ms"
        )

    def get_correlations_for_market(
        self,
        market_id: str
    ) -> List[MarketCorrelation]:
        """Get all correlations involving a specific market."""
        corr_ids = self.market_to_correlations.get(market_id, [])
        return [self.correlations[cid] for cid in corr_ids if cid in self.correlations]

    def prioritize_markets(
        self,
        active_signals: Dict[str, ScoredSignal]
    ) -> List[str]:
        """
        Determine which markets need most frequent updates.

        Priority based on:
        - Markets with high-scoring active signals
        - Markets with arbitrage opportunities
        - Markets with recent large price moves
        - Markets with pending resolution
        """
        # Update priorities based on active signals
        self._update_priorities_from_signals(active_signals)

        # Sort markets by priority score
        sorted_markets = sorted(
            self.market_priorities.items(),
            key=lambda x: x[1].priority_score,
            reverse=True
        )

        return [market_id for market_id, _ in sorted_markets]

    def _update_priorities_from_signals(
        self,
        active_signals: Dict[str, ScoredSignal]
    ) -> None:
        """Update market priorities based on active signals."""
        # Reset counts
        for priority in self.market_priorities.values():
            priority.active_signal_count = 0
            priority.has_arbitrage = False

        # Count signals per market
        market_signal_counts: Dict[str, int] = defaultdict(int)
        market_max_scores: Dict[str, float] = defaultdict(float)
        market_has_arbitrage: Dict[str, bool] = defaultdict(bool)

        for signal in active_signals.values():
            for market_id in signal.divergence.market_ids:
                market_signal_counts[market_id] += 1
                market_max_scores[market_id] = max(
                    market_max_scores[market_id],
                    signal.overall_score
                )
                if signal.divergence.is_arbitrage:
                    market_has_arbitrage[market_id] = True

        # Update priorities
        for market_id in set(market_signal_counts.keys()) | set(self.market_priorities.keys()):
            if market_id not in self.market_priorities:
                self.market_priorities[market_id] = MarketPriority(
                    market_id=market_id,
                    priority_score=0.0,
                    last_update=datetime.utcnow(),
                    active_signal_count=0,
                )

            priority = self.market_priorities[market_id]
            priority.active_signal_count = market_signal_counts.get(market_id, 0)
            priority.has_arbitrage = market_has_arbitrage.get(market_id, False)

            # Calculate priority score
            score = 0.0

            # Base score from signal count
            score += priority.active_signal_count * 2.0

            # Boost for arbitrage
            if priority.has_arbitrage:
                score += self.config.arbitrage_priority_boost

            # Boost for high-scoring signals
            max_signal_score = market_max_scores.get(market_id, 0)
            if max_signal_score >= self.config.high_score_threshold:
                score += self.config.high_score_priority_boost

            # Boost for recent price moves
            if priority.recent_price_move >= self.config.recent_move_threshold:
                score += self.config.recent_move_priority_boost

            priority.priority_score = score

        # Update hot markets
        self.hot_markets = {
            market_id for market_id, priority in self.market_priorities.items()
            if priority.priority_score >= 5.0 or priority.has_arbitrage
        }

    def record_price_move(
        self,
        market_id: str,
        price_change_pct: float
    ) -> None:
        """Record a price move for priority calculation."""
        now = datetime.utcnow()
        self.recent_moves[market_id].append((now, abs(price_change_pct)))

        # Keep only recent moves (last 5 minutes)
        cutoff = now - timedelta(minutes=5)
        self.recent_moves[market_id] = [
            (t, p) for t, p in self.recent_moves[market_id]
            if t > cutoff
        ]

        # Update market priority
        if market_id in self.market_priorities:
            # Max recent move
            max_move = max(p for _, p in self.recent_moves[market_id]) if self.recent_moves[market_id] else 0
            self.market_priorities[market_id].recent_price_move = max_move

    def batch_correlations(
        self,
        correlations: List[MarketCorrelation] = None,
        batch_size: int = None
    ) -> Iterator[List[MarketCorrelation]]:
        """
        Split correlations into batches for parallel processing.

        Yields batches of correlations.
        """
        if correlations is None:
            correlations = list(self.correlations.values())

        if batch_size is None:
            batch_size = self.config.default_batch_size

        batch_size = min(batch_size, self.config.max_batch_size)

        for i in range(0, len(correlations), batch_size):
            yield correlations[i:i + batch_size]

    def batch_by_priority(
        self,
        correlations: List[MarketCorrelation] = None,
        batch_size: int = None
    ) -> Iterator[List[MarketCorrelation]]:
        """
        Batch correlations with priority-based ordering.

        Higher priority correlations come first.
        """
        if correlations is None:
            correlations = list(self.correlations.values())

        if batch_size is None:
            batch_size = self.config.default_batch_size

        # Score each correlation
        def correlation_priority(corr: MarketCorrelation) -> float:
            score = 0.0
            for market_id in [corr.market_a_id, corr.market_b_id]:
                if market_id in self.market_priorities:
                    score += self.market_priorities[market_id].priority_score
            return score

        # Sort by priority
        sorted_corrs = sorted(correlations, key=correlation_priority, reverse=True)

        # Yield batches
        for i in range(0, len(sorted_corrs), batch_size):
            yield sorted_corrs[i:i + batch_size]

    def get_update_interval(self, market_id: str) -> float:
        """Get recommended update interval for a market."""
        if market_id in self.hot_markets:
            return self.config.hot_market_update_interval
        return self.config.normal_market_update_interval

    def should_check_correlation(
        self,
        correlation: MarketCorrelation,
        last_check: Optional[datetime] = None
    ) -> bool:
        """
        Determine if a correlation should be checked now.

        Based on priority and time since last check.
        """
        if last_check is None:
            return True

        # Get priority for this correlation
        priority = 0.0
        for market_id in [correlation.market_a_id, correlation.market_b_id]:
            if market_id in self.market_priorities:
                priority += self.market_priorities[market_id].priority_score

        # Determine interval based on priority
        if priority >= 10:
            interval = 1.0  # Check every second
        elif priority >= 5:
            interval = 2.0
        elif priority >= 1:
            interval = 5.0
        else:
            interval = 10.0

        time_since_check = (datetime.utcnow() - last_check).total_seconds()
        return time_since_check >= interval

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_correlations": self._correlation_count,
            "markets_indexed": len(self.market_to_correlations),
            "hot_markets": len(self.hot_markets),
            "markets_with_priority": len(self.market_priorities),
            "index_build_time": self._index_build_time.isoformat() if self._index_build_time else None,
            "avg_correlations_per_market": (
                sum(len(corrs) for corrs in self.market_to_correlations.values()) /
                len(self.market_to_correlations)
                if self.market_to_correlations else 0
            ),
        }


class CorrelationBatcher:
    """
    Handles efficient batching of correlation checks.

    Supports:
    - Fixed-size batching
    - Priority-based batching
    - Adaptive batch sizing based on latency
    """

    def __init__(
        self,
        target_latency_ms: float = 100.0,
        initial_batch_size: int = 50
    ):
        self.target_latency_ms = target_latency_ms
        self.batch_size = initial_batch_size
        self.min_batch_size = 10
        self.max_batch_size = 200

        # Latency history for adaptive sizing
        self._latencies: List[float] = []
        self._max_history = 100

    def get_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.batch_size

    def record_batch_latency(
        self,
        batch_size: int,
        latency_ms: float
    ) -> None:
        """
        Record latency for a batch processing.

        Adjusts batch size based on latency trends.
        """
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._max_history:
            self._latencies = self._latencies[-self._max_history:]

        # Adjust batch size based on recent latencies
        if len(self._latencies) >= 5:
            avg_latency = sum(self._latencies[-5:]) / 5

            if avg_latency > self.target_latency_ms * 1.5:
                # Too slow, reduce batch size
                self.batch_size = max(
                    self.min_batch_size,
                    int(self.batch_size * 0.8)
                )
            elif avg_latency < self.target_latency_ms * 0.5:
                # Too fast, can increase batch size
                self.batch_size = min(
                    self.max_batch_size,
                    int(self.batch_size * 1.2)
                )

    def create_batches(
        self,
        items: List[Any],
        batch_size: int = None
    ) -> List[List[Any]]:
        """Create batches from items."""
        if batch_size is None:
            batch_size = self.batch_size

        return [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

"""
Signal aggregation pipeline.

Combines filtering, blacklist/whitelist, ranking, and rate limiting
into a single processing pipeline.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Tuple

from src.signals.scoring.types import ScoredSignal
from src.signals.filtering.filter import (
    SignalFilter,
    FilterConfig,
    FilterDecision,
    FilterResult,
)
from src.signals.filtering.blacklist import (
    SignalBlacklist,
    SignalWhitelist,
)
from src.signals.filtering.ranker import (
    SignalRanker,
    RankingConfig,
    RankedSignal,
    SignalTier,
)
from src.signals.filtering.rate_limiter import (
    SignalRateLimiter,
    RateLimitConfig,
    RateLimitResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for the signal aggregator."""
    # Pipeline stages to enable
    enable_filtering: bool = True
    enable_blacklist: bool = True
    enable_whitelist: bool = True
    enable_ranking: bool = True
    enable_rate_limiting: bool = True

    # Output limits
    max_output_signals: int = 50
    min_tier_for_output: SignalTier = SignalTier.LOW

    # Configs for sub-components (if not provided, use defaults)
    filter_config: Optional[FilterConfig] = None
    ranking_config: Optional[RankingConfig] = None
    rate_limit_config: Optional[RateLimitConfig] = None


@dataclass
class PipelineStage:
    """Information about a pipeline stage."""
    name: str
    enabled: bool
    input_count: int
    output_count: int
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatorResult:
    """Result of signal aggregation."""
    # Output signals
    signals: List[RankedSignal]

    # Pipeline metadata
    stages: List[PipelineStage]
    total_duration_ms: float
    input_count: int
    output_count: int

    # Summary
    by_tier: Dict[SignalTier, int] = field(default_factory=dict)


class SignalAggregator:
    """
    Full signal processing pipeline.

    Pipeline stages:
    1. Blacklist filtering (remove blocked signals)
    2. Quality filtering (apply filter criteria)
    3. Whitelist boosting (apply score boosts)
    4. Ranking (sort by actionability)
    5. Rate limiting (prevent alert fatigue)
    6. Output limiting (cap total signals)
    """

    def __init__(self, config: AggregatorConfig = None):
        self.config = config or AggregatorConfig()

        # Initialize components
        self.filter = SignalFilter(self.config.filter_config)
        self.blacklist = SignalBlacklist()
        self.whitelist = SignalWhitelist()
        self.ranker = SignalRanker(self.config.ranking_config)
        self.rate_limiter = SignalRateLimiter(self.config.rate_limit_config)

        # Custom pipeline stages
        self._custom_stages: List[Tuple[str, Callable[[List[ScoredSignal]], List[ScoredSignal]]]] = []

        # Statistics
        self._total_processed = 0
        self._total_output = 0

    def process(self, signals: List[ScoredSignal]) -> AggregatorResult:
        """
        Process signals through the full pipeline.

        Returns AggregatorResult with processed signals and metadata.
        """
        start_time = datetime.utcnow()
        stages: List[PipelineStage] = []
        current_signals = signals
        input_count = len(signals)

        self._total_processed += input_count

        # Stage 1: Blacklist filtering
        if self.config.enable_blacklist:
            stage_start = datetime.utcnow()
            current_signals = self._apply_blacklist(current_signals)
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name="blacklist",
                enabled=True,
                input_count=input_count,
                output_count=len(current_signals),
                duration_ms=stage_duration,
                details={"blocked": input_count - len(current_signals)},
            ))

        # Stage 2: Quality filtering
        if self.config.enable_filtering:
            stage_start = datetime.utcnow()
            pre_count = len(current_signals)
            current_signals = self.filter.filter(current_signals)
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name="filter",
                enabled=True,
                input_count=pre_count,
                output_count=len(current_signals),
                duration_ms=stage_duration,
                details={"filtered_out": pre_count - len(current_signals)},
            ))

        # Stage 3: Custom stages
        for stage_name, stage_func in self._custom_stages:
            stage_start = datetime.utcnow()
            pre_count = len(current_signals)
            try:
                current_signals = stage_func(current_signals)
            except Exception as e:
                logger.error(f"Custom stage {stage_name} error: {e}")
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name=stage_name,
                enabled=True,
                input_count=pre_count,
                output_count=len(current_signals),
                duration_ms=stage_duration,
            ))

        # Stage 4: Whitelist boost calculation
        whitelist_boosts: Dict[str, float] = {}
        if self.config.enable_whitelist:
            stage_start = datetime.utcnow()
            for signal in current_signals:
                boost, _ = self.whitelist.get_boost(signal)
                if boost > 0:
                    for market_id in signal.divergence.market_ids:
                        whitelist_boosts[market_id] = boost
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name="whitelist",
                enabled=True,
                input_count=len(current_signals),
                output_count=len(current_signals),
                duration_ms=stage_duration,
                details={"boosted_markets": len(whitelist_boosts)},
            ))

        # Stage 5: Ranking
        ranked_signals: List[RankedSignal] = []
        if self.config.enable_ranking:
            stage_start = datetime.utcnow()
            ranked_signals = self.ranker.rank(current_signals, whitelist_boosts)
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name="ranking",
                enabled=True,
                input_count=len(current_signals),
                output_count=len(ranked_signals),
                duration_ms=stage_duration,
            ))
        else:
            # If ranking disabled, create basic RankedSignal wrappers
            ranked_signals = [
                RankedSignal(
                    signal=s,
                    rank=i + 1,
                    tier=SignalTier.MEDIUM,
                    effective_score=s.overall_score,
                )
                for i, s in enumerate(current_signals)
            ]

        # Stage 6: Tier filtering
        tier_order = [
            SignalTier.CRITICAL,
            SignalTier.HIGH,
            SignalTier.MEDIUM,
            SignalTier.LOW,
            SignalTier.IGNORE,
        ]
        min_tier_idx = tier_order.index(self.config.min_tier_for_output)
        allowed_tiers = set(tier_order[:min_tier_idx + 1])

        pre_tier_count = len(ranked_signals)
        ranked_signals = [r for r in ranked_signals if r.tier in allowed_tiers]
        stages.append(PipelineStage(
            name="tier_filter",
            enabled=True,
            input_count=pre_tier_count,
            output_count=len(ranked_signals),
            duration_ms=0,  # Negligible
            details={"min_tier": self.config.min_tier_for_output.value},
        ))

        # Stage 7: Rate limiting
        if self.config.enable_rate_limiting:
            stage_start = datetime.utcnow()
            pre_count = len(ranked_signals)
            filtered_ranked = []
            for ranked in ranked_signals:
                result = self.rate_limiter.check(ranked.signal)
                if result.allowed:
                    self.rate_limiter.record(ranked.signal)
                    filtered_ranked.append(ranked)
            ranked_signals = filtered_ranked
            stage_duration = (datetime.utcnow() - stage_start).total_seconds() * 1000
            stages.append(PipelineStage(
                name="rate_limit",
                enabled=True,
                input_count=pre_count,
                output_count=len(ranked_signals),
                duration_ms=stage_duration,
                details={"rate_limited": pre_count - len(ranked_signals)},
            ))

        # Stage 8: Output limiting
        if len(ranked_signals) > self.config.max_output_signals:
            ranked_signals = ranked_signals[:self.config.max_output_signals]

        # Calculate tier distribution
        by_tier: Dict[SignalTier, int] = {}
        for tier in SignalTier:
            count = sum(1 for r in ranked_signals if r.tier == tier)
            if count > 0:
                by_tier[tier] = count

        total_duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        self._total_output += len(ranked_signals)

        return AggregatorResult(
            signals=ranked_signals,
            stages=stages,
            total_duration_ms=total_duration,
            input_count=input_count,
            output_count=len(ranked_signals),
            by_tier=by_tier,
        )

    def _apply_blacklist(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """Apply blacklist filtering."""
        result = []
        for signal in signals:
            is_blocked, reason = self.blacklist.is_blocked(signal)
            if not is_blocked:
                result.append(signal)
            else:
                logger.debug(f"Signal blocked: {reason}")
        return result

    def add_custom_stage(
        self,
        name: str,
        stage_func: Callable[[List[ScoredSignal]], List[ScoredSignal]]
    ) -> None:
        """
        Add a custom pipeline stage.

        Custom stages run after filtering but before ranking.
        """
        self._custom_stages.append((name, stage_func))

    def remove_custom_stage(self, name: str) -> bool:
        """Remove a custom stage by name."""
        for i, (stage_name, _) in enumerate(self._custom_stages):
            if stage_name == name:
                del self._custom_stages[i]
                return True
        return False

    def get_actionable(self, signals: List[ScoredSignal]) -> List[RankedSignal]:
        """
        Convenience method to get only actionable signals.

        Returns signals with tier HIGH or CRITICAL.
        """
        result = self.process(signals)
        return [
            r for r in result.signals
            if r.tier in {SignalTier.CRITICAL, SignalTier.HIGH}
        ]

    def get_top_n(
        self,
        signals: List[ScoredSignal],
        n: int
    ) -> List[RankedSignal]:
        """
        Get top N signals after full processing.
        """
        result = self.process(signals)
        return result.signals[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_processed": self._total_processed,
            "total_output": self._total_output,
            "output_rate": (
                self._total_output / self._total_processed
                if self._total_processed > 0 else 0
            ),
            "filter_stats": self.filter.get_stats(),
            "blacklist_stats": self.blacklist.get_stats(),
            "whitelist_stats": self.whitelist.get_stats(),
            "ranker_stats": self.ranker.get_stats(),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._total_processed = 0
        self._total_output = 0
        self.filter.reset_stats()
        self.ranker.reset_stats()
        self.rate_limiter.reset()


class StreamingAggregator:
    """
    Streaming version of SignalAggregator.

    Processes signals one at a time, suitable for real-time streams.
    """

    def __init__(self, config: AggregatorConfig = None):
        self.config = config or AggregatorConfig()
        self.filter = SignalFilter(self.config.filter_config)
        self.blacklist = SignalBlacklist()
        self.whitelist = SignalWhitelist()
        self.ranker = SignalRanker(self.config.ranking_config)
        self.rate_limiter = SignalRateLimiter(self.config.rate_limit_config)

        # Buffer for batch ranking
        self._buffer: List[ScoredSignal] = []
        self._buffer_size = 10
        self._last_flush = datetime.utcnow()
        self._flush_interval_seconds = 5.0

    def submit(self, signal: ScoredSignal) -> Optional[RankedSignal]:
        """
        Submit a single signal for processing.

        Returns RankedSignal if signal passes all checks, None otherwise.
        """
        # Quick reject via blacklist
        if self.config.enable_blacklist:
            is_blocked, _ = self.blacklist.is_blocked(signal)
            if is_blocked:
                return None

        # Quality filter
        if self.config.enable_filtering:
            decisions = self.filter.evaluate(signal)
            if any(d.result == FilterResult.REJECT for d in decisions):
                return None

        # Rate limit check
        if self.config.enable_rate_limiting:
            result = self.rate_limiter.check(signal)
            if not result.allowed:
                return None
            self.rate_limiter.record(signal)

        # Get whitelist boost
        boost, _ = self.whitelist.get_boost(signal)

        # Create ranked signal
        effective_score = signal.overall_score + boost

        # Determine tier
        if effective_score >= 85:
            tier = SignalTier.CRITICAL
        elif effective_score >= 70:
            tier = SignalTier.HIGH
        elif effective_score >= 50:
            tier = SignalTier.MEDIUM
        elif effective_score >= 30:
            tier = SignalTier.LOW
        else:
            tier = SignalTier.IGNORE

        # Check minimum tier
        tier_order = [
            SignalTier.CRITICAL,
            SignalTier.HIGH,
            SignalTier.MEDIUM,
            SignalTier.LOW,
            SignalTier.IGNORE,
        ]
        if tier_order.index(tier) > tier_order.index(self.config.min_tier_for_output):
            return None

        return RankedSignal(
            signal=signal,
            rank=0,  # Not meaningful for streaming
            tier=tier,
            effective_score=effective_score,
            rank_factors={"whitelist_boost": boost} if boost > 0 else {},
        )

    def submit_batch(self, signals: List[ScoredSignal]) -> List[RankedSignal]:
        """Submit a batch of signals."""
        results = []
        for signal in signals:
            ranked = self.submit(signal)
            if ranked is not None:
                results.append(ranked)
        return results

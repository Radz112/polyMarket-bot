"""
Signal filtering and ranking module.

Provides noise reduction, prioritization, and formatting for trading signals.
"""
from src.signals.filtering.filter import (
    SignalFilter,
    FilterConfig,
    FilterResult,
    FilterDecision,
    CompositeFilter,
)
from src.signals.filtering.blacklist import (
    SignalBlacklist,
    SignalWhitelist,
    BlockReason,
    BoostReason,
    BlockEntry,
    BoostEntry,
)
from src.signals.filtering.ranker import (
    SignalRanker,
    RankingConfig,
    RankedSignal,
    SignalTier,
    MultiFactorRanker,
)
from src.signals.filtering.rate_limiter import (
    SignalRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
    SlidingWindowCounter,
    TokenBucket,
)
from src.signals.filtering.aggregator import (
    SignalAggregator,
    AggregatorConfig,
    AggregatorResult,
    PipelineStage,
    StreamingAggregator,
)
from src.signals.filtering.formatter import (
    SignalFormatter,
    FormatterConfig,
    OutputFormat,
    AlertFormatter,
)

__all__ = [
    # Filter
    "SignalFilter",
    "FilterConfig",
    "FilterResult",
    "FilterDecision",
    "CompositeFilter",
    # Blacklist/Whitelist
    "SignalBlacklist",
    "SignalWhitelist",
    "BlockReason",
    "BoostReason",
    "BlockEntry",
    "BoostEntry",
    # Ranker
    "SignalRanker",
    "RankingConfig",
    "RankedSignal",
    "SignalTier",
    "MultiFactorRanker",
    # Rate Limiter
    "SignalRateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitStrategy",
    "SlidingWindowCounter",
    "TokenBucket",
    # Aggregator
    "SignalAggregator",
    "AggregatorConfig",
    "AggregatorResult",
    "PipelineStage",
    "StreamingAggregator",
    # Formatter
    "SignalFormatter",
    "FormatterConfig",
    "OutputFormat",
    "AlertFormatter",
]

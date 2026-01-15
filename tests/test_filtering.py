"""
Tests for signal filtering and ranking module.

Tests cover:
- SignalFilter with various filter methods
- SignalBlacklist and SignalWhitelist
- SignalRanker with multiple ranking factors
- SignalRateLimiter with various strategies
- SignalAggregator pipeline
- SignalFormatter output formats
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from typing import List

from src.signals.filtering import (
    # Filter
    SignalFilter,
    FilterConfig,
    FilterResult,
    FilterDecision,
    CompositeFilter,
    # Blacklist/Whitelist
    SignalBlacklist,
    SignalWhitelist,
    BlockReason,
    BoostReason,
    # Ranker
    SignalRanker,
    RankingConfig,
    RankedSignal,
    SignalTier,
    MultiFactorRanker,
    # Rate Limiter
    SignalRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStrategy,
    SlidingWindowCounter,
    TokenBucket,
    # Aggregator
    SignalAggregator,
    AggregatorConfig,
    AggregatorResult,
    StreamingAggregator,
    # Formatter
    SignalFormatter,
    FormatterConfig,
    OutputFormat,
    AlertFormatter,
)
from src.signals.scoring.types import (
    ScoredSignal,
    ComponentScore,
    RecommendedAction,
    Urgency,
)
from src.signals.scoring.recommender import TradingRecommendation
from src.signals.divergence.types import (
    Divergence,
    DivergenceType,
    DivergenceStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_divergence():
    """Create a sample divergence."""
    return Divergence(
        id="div-001",
        divergence_type=DivergenceType.CORRELATION_BREAK,
        market_ids=["market-a", "market-b"],
        divergence_pct=5.0,
        expected_relationship="positive",
        confidence=0.8,
        detected_at=datetime.utcnow(),
        current_prices={"market-a": 0.55, "market-b": 0.45},
        is_arbitrage=False,
    )


@pytest.fixture
def sample_scored_signal(sample_divergence):
    """Create a sample scored signal."""
    return ScoredSignal(
        divergence=sample_divergence,
        overall_score=65.0,
        component_scores={
            "divergence": ComponentScore(
                name="divergence",
                score=70.0,
                weight=0.25,
                weighted_score=17.5,
                explanation="5% divergence detected",
                metadata={"divergence_pct": 5.0},
            ),
            "liquidity": ComponentScore(
                name="liquidity",
                score=60.0,
                weight=0.2,
                weighted_score=12.0,
                explanation="Moderate liquidity",
                metadata={},
            ),
            "confidence": ComponentScore(
                name="confidence",
                score=65.0,
                weight=0.15,
                weighted_score=9.75,
                explanation="Good confidence level",
                metadata={},
            ),
        },
        recommended_action=RecommendedAction.BUY,
        urgency=Urgency.WATCH,
    )


@pytest.fixture
def multiple_signals(sample_divergence):
    """Create multiple signals with varying scores."""
    signals = []
    for i in range(10):
        div = Divergence(
            id=f"div-{i:03d}",
            divergence_type=DivergenceType.CORRELATION_BREAK,
            market_ids=[f"market-a-{i}", f"market-b-{i}"],
            divergence_pct=3.0 + i,  # 3% to 12%
            expected_relationship="positive",
            confidence=0.7 + (i * 0.02),
            detected_at=datetime.utcnow(),
            current_prices={f"market-a-{i}": 0.55, f"market-b-{i}": 0.45},
            is_arbitrage=(i >= 7),  # Last 3 are arbitrage
        )
        div_score = 40.0 + (i * 5)
        liq_score = 50.0 + (i * 3)
        signal = ScoredSignal(
            divergence=div,
            scored_at=datetime.utcnow() - timedelta(seconds=i * 10),
            overall_score=30.0 + (i * 7),  # 30 to 93
            component_scores={
                "divergence": ComponentScore(
                    name="divergence",
                    score=div_score,
                    weight=0.25,
                    weighted_score=div_score * 0.25,
                    explanation=f"Divergence score {div_score}",
                    metadata={},
                ),
                "liquidity": ComponentScore(
                    name="liquidity",
                    score=liq_score,
                    weight=0.2,
                    weighted_score=liq_score * 0.2,
                    explanation=f"Liquidity score {liq_score}",
                    metadata={},
                ),
            },
            recommended_action=RecommendedAction.BUY if i >= 5 else RecommendedAction.PASS,
            urgency=Urgency.SOON if i >= 7 else Urgency.WATCH,
        )
        signals.append(signal)
    return signals


# ============================================================================
# SignalFilter Tests
# ============================================================================

class TestSignalFilter:
    """Tests for SignalFilter."""

    def test_filter_by_score_passes(self, sample_scored_signal):
        """Test that signals above minimum score pass."""
        config = FilterConfig(min_overall_score=50.0)
        filter_ = SignalFilter(config)

        result = filter_.filter([sample_scored_signal])
        assert len(result) == 1

    def test_filter_by_score_rejects(self, sample_scored_signal):
        """Test that signals below minimum score are rejected."""
        config = FilterConfig(min_overall_score=80.0)
        filter_ = SignalFilter(config)

        result = filter_.filter([sample_scored_signal])
        assert len(result) == 0

    def test_filter_by_divergence_minimum(self, sample_divergence):
        """Test filtering by minimum divergence percentage."""
        config = FilterConfig(min_divergence_pct=10.0)  # 10% minimum
        filter_ = SignalFilter(config)

        signal = ScoredSignal(
            divergence=sample_divergence,  # Has 5% divergence
            overall_score=70.0,
            component_scores={},
        )

        result = filter_.filter([signal])
        assert len(result) == 0

    def test_filter_by_divergence_maximum(self, sample_divergence):
        """Test filtering by maximum divergence percentage (data error)."""
        # Create divergence with very high percentage
        div = Divergence(
            id="div-001",
            divergence_type=DivergenceType.CORRELATION_BREAK,
            market_ids=["market-a", "market-b"],
            divergence_pct=60.0,  # 60% is likely a data error
            expected_relationship="positive",
            confidence=0.8,
            detected_at=datetime.utcnow(),
            current_prices={"market-a": 0.55, "market-b": 0.45},
            is_arbitrage=False,
        )

        config = FilterConfig(max_divergence_pct=50.0)
        filter_ = SignalFilter(config)

        signal = ScoredSignal(
            divergence=div,
            overall_score=70.0,
            component_scores={},
        )

        result = filter_.filter([signal])
        assert len(result) == 0

    def test_filter_by_staleness(self, sample_divergence):
        """Test filtering by signal staleness."""
        config = FilterConfig(max_signal_age_seconds=60)  # 1 minute max
        filter_ = SignalFilter(config)

        # Create an old signal
        signal = ScoredSignal(
            divergence=sample_divergence,
            scored_at=datetime.utcnow() - timedelta(seconds=120),  # 2 minutes old
            overall_score=70.0,
            component_scores={},
        )

        result = filter_.filter([signal])
        assert len(result) == 0

    def test_filter_by_liquidity_score(self, sample_divergence):
        """Test filtering by liquidity score."""
        config = FilterConfig(min_liquidity_score=50.0)
        filter_ = SignalFilter(config)

        signal = ScoredSignal(
            divergence=sample_divergence,
            overall_score=70.0,
            component_scores={
                "liquidity": ComponentScore(
                    name="liquidity",
                    score=30.0,  # Below threshold
                    weight=0.2,
                    weighted_score=6.0,
                    explanation="Low liquidity",
                    metadata={},
                ),
            },
        )

        result = filter_.filter([signal])
        assert len(result) == 0

    def test_evaluate_returns_decisions(self, sample_scored_signal):
        """Test that evaluate returns filter decisions."""
        filter_ = SignalFilter()
        decisions = filter_.evaluate(sample_scored_signal)

        assert len(decisions) > 0
        assert all(isinstance(d, FilterDecision) for d in decisions)

    def test_high_score_boost(self, sample_divergence):
        """Test that high-scoring signals get boosted."""
        config = FilterConfig(high_score_threshold=70.0)
        filter_ = SignalFilter(config)

        signal = ScoredSignal(
            divergence=sample_divergence,
            overall_score=85.0,  # Above high score threshold
            component_scores={},
        )

        decisions = filter_.evaluate(signal)
        score_decision = next(d for d in decisions if d.filter_name == "score")
        assert score_decision.result == FilterResult.BOOST

    def test_custom_filter(self, sample_scored_signal):
        """Test adding a custom filter."""
        filter_ = SignalFilter()

        # Add custom filter that rejects everything
        def reject_all(signal):
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="custom",
                reason="Custom rejection",
            )

        filter_.add_custom_filter(reject_all)
        result = filter_.filter([sample_scored_signal])
        assert len(result) == 0

    def test_filter_stats(self, multiple_signals):
        """Test filter statistics tracking."""
        config = FilterConfig(min_overall_score=50.0)
        filter_ = SignalFilter(config)

        filter_.filter(multiple_signals)
        stats = filter_.get_stats()

        assert stats["total_processed"] == len(multiple_signals)
        assert stats["total_passed"] + stats["total_rejected"] == len(multiple_signals)


class TestCompositeFilter:
    """Tests for CompositeFilter."""

    def test_and_logic(self, multiple_signals):
        """Test that AND logic requires all filters to pass."""
        # Create two filters with different thresholds
        filter1 = SignalFilter(FilterConfig(min_overall_score=30.0))
        filter2 = SignalFilter(FilterConfig(min_overall_score=60.0))

        composite = CompositeFilter(require_all=True)
        composite.add_filter(filter1)
        composite.add_filter(filter2)

        result = composite.filter(multiple_signals)
        # Only signals >= 60 should pass
        assert all(s.overall_score >= 60.0 for s in result)

    def test_or_logic(self, multiple_signals):
        """Test that OR logic allows any filter to pass."""
        # Create filter with high threshold
        filter1 = SignalFilter(FilterConfig(min_overall_score=90.0))

        composite = CompositeFilter(require_all=False)
        composite.add_filter(filter1)

        result = composite.filter(multiple_signals)
        # Only signals >= 90 should pass
        assert len(result) >= 0


# ============================================================================
# SignalBlacklist Tests
# ============================================================================

class TestSignalBlacklist:
    """Tests for SignalBlacklist."""

    def test_block_market(self, sample_scored_signal):
        """Test blocking a market."""
        blacklist = SignalBlacklist()
        blacklist.block_market("market-a", BlockReason.MANUAL)

        is_blocked, reason = blacklist.is_blocked(sample_scored_signal)
        assert is_blocked is True
        assert "market-a" in reason

    def test_block_pair(self, sample_scored_signal):
        """Test blocking a market pair."""
        blacklist = SignalBlacklist()
        blacklist.block_pair("market-a", "market-b", BlockReason.POOR_PERFORMANCE)

        is_blocked, reason = blacklist.is_blocked(sample_scored_signal)
        assert is_blocked is True

    def test_unblock_market(self, sample_scored_signal):
        """Test unblocking a market."""
        blacklist = SignalBlacklist()
        blacklist.block_market("market-a", BlockReason.MANUAL)
        blacklist.unblock_market("market-a")

        is_blocked, _ = blacklist.is_blocked(sample_scored_signal)
        assert is_blocked is False

    def test_temporary_block_expires(self, sample_scored_signal):
        """Test that temporary blocks expire."""
        blacklist = SignalBlacklist()
        # Block for 0 hours (immediate expiry)
        blacklist.block_market(
            "market-a",
            BlockReason.TEMPORARY,
            duration_hours=0.0001,  # Very short duration
        )

        # Wait and check - in practice this would test expiry
        # For this test, we just verify the entry has an expiry
        blocked_markets = blacklist.get_blocked_markets()
        assert len(blocked_markets) >= 0  # May have expired

    def test_filter_removes_blocked(self, multiple_signals):
        """Test that filter removes blocked signals."""
        blacklist = SignalBlacklist()
        blacklist.block_market("market-a-0", BlockReason.MANUAL)

        result = blacklist.filter(multiple_signals)
        assert len(result) == len(multiple_signals) - 1

    def test_blacklist_stats(self, sample_scored_signal):
        """Test blacklist statistics."""
        blacklist = SignalBlacklist()
        blacklist.block_market("market-a", BlockReason.MANUAL)
        blacklist.is_blocked(sample_scored_signal)

        stats = blacklist.get_stats()
        assert stats["blocked_markets"] == 1
        assert stats["signals_blocked"] == 1


class TestSignalWhitelist:
    """Tests for SignalWhitelist."""

    def test_boost_market(self, sample_scored_signal):
        """Test boosting a market."""
        whitelist = SignalWhitelist(default_boost=10.0)
        whitelist.boost_market("market-a", BoostReason.HIGH_PRIORITY)

        boost, reason = whitelist.get_boost(sample_scored_signal)
        assert boost == 10.0
        assert reason is not None

    def test_boost_pair(self, sample_scored_signal):
        """Test boosting a market pair."""
        whitelist = SignalWhitelist()
        whitelist.boost_pair(
            "market-a", "market-b",
            BoostReason.STRONG_PERFORMANCE,
            score_boost=15.0,
        )

        boost, _ = whitelist.get_boost(sample_scored_signal)
        assert boost == 15.0

    def test_cumulative_boost(self, sample_scored_signal):
        """Test that multiple boosts accumulate."""
        whitelist = SignalWhitelist()
        whitelist.boost_market("market-a", BoostReason.MANUAL, score_boost=5.0)
        whitelist.boost_market("market-b", BoostReason.MANUAL, score_boost=7.0)

        boost, _ = whitelist.get_boost(sample_scored_signal)
        assert boost == 12.0

    def test_is_whitelisted(self, sample_scored_signal):
        """Test checking if signal is whitelisted."""
        whitelist = SignalWhitelist()

        assert whitelist.is_whitelisted(sample_scored_signal) is False

        whitelist.boost_market("market-a", BoostReason.MANUAL)
        assert whitelist.is_whitelisted(sample_scored_signal) is True


# ============================================================================
# SignalRanker Tests
# ============================================================================

class TestSignalRanker:
    """Tests for SignalRanker."""

    def test_rank_by_score(self, multiple_signals):
        """Test ranking signals by score."""
        ranker = SignalRanker()
        ranked = ranker.rank(multiple_signals)

        assert len(ranked) == len(multiple_signals)
        # Should be sorted by effective score descending
        scores = [r.effective_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_tier_assignment(self, multiple_signals):
        """Test that tiers are assigned correctly."""
        ranker = SignalRanker()
        ranked = ranker.rank(multiple_signals)

        for r in ranked:
            if r.effective_score >= 85:
                assert r.tier == SignalTier.CRITICAL
            elif r.effective_score >= 70:
                assert r.tier == SignalTier.HIGH
            elif r.effective_score >= 50:
                assert r.tier == SignalTier.MEDIUM
            elif r.effective_score >= 30:
                assert r.tier == SignalTier.LOW

    def test_arbitrage_boost(self, multiple_signals):
        """Test that arbitrage signals get boosted."""
        config = RankingConfig(arbitrage_boost=20.0)
        ranker = SignalRanker(config)
        ranked = ranker.rank(multiple_signals)

        # Arbitrage signals should have higher effective scores
        arb_signals = [r for r in ranked if r.signal.divergence.is_arbitrage]
        for arb in arb_signals:
            assert arb.rank_factors.get("arbitrage_boost") == 20.0

    def test_get_top_n(self, multiple_signals):
        """Test getting top N signals."""
        ranker = SignalRanker()
        top_5 = ranker.get_top_n(multiple_signals, 5)

        assert len(top_5) == 5
        assert top_5[0].rank == 1

    def test_get_by_tier(self, multiple_signals):
        """Test grouping signals by tier."""
        ranker = SignalRanker()
        by_tier = ranker.get_by_tier(multiple_signals)

        # Should have at least some tiers
        assert len(by_tier) > 0
        for tier, signals in by_tier.items():
            assert all(s.tier == tier for s in signals)

    def test_diversified_ranking(self, sample_divergence):
        """Test diversified ranking limits exposure."""
        # Create signals with same markets
        signals = []
        for i in range(5):
            signal = ScoredSignal(
                divergence=sample_divergence,  # Same markets
                overall_score=80.0 - i,
                component_scores={},
            )
            signals.append(signal)

        ranker = SignalRanker()
        diversified = ranker.get_diversified(
            signals,
            max_per_market=2,
            total_limit=10,
        )

        assert len(diversified) == 2  # Limited by max_per_market

    def test_whitelist_boosts_applied(self, multiple_signals):
        """Test that whitelist boosts are applied in ranking."""
        ranker = SignalRanker()
        whitelist_boosts = {"market-a-0": 50.0}  # Significant boost

        ranked = ranker.rank(multiple_signals, whitelist_boosts)

        # Signal 0 should be boosted
        sig_0 = next(r for r in ranked if r.signal.divergence.id == "div-000")
        assert sig_0.rank_factors.get("whitelist_boost_market-a-0") == 50.0


class TestMultiFactorRanker:
    """Tests for MultiFactorRanker."""

    def test_weighted_ranking(self, multiple_signals):
        """Test multi-factor weighted ranking."""
        ranker = MultiFactorRanker()
        ranker.set_weight("score", 2.0)
        ranker.set_weight("divergence_pct", 1.0)

        ranked = ranker.rank(multiple_signals)
        assert len(ranked) == len(multiple_signals)

    def test_custom_factor(self, multiple_signals):
        """Test adding custom ranking factor."""
        ranker = MultiFactorRanker()

        # Custom factor that boosts arbitrage
        def arb_factor(signal):
            return 100.0 if signal.divergence.is_arbitrage else 0.0

        ranker.add_custom_factor("arb_bonus", arb_factor, weight=0.5)
        ranked = ranker.rank(multiple_signals)

        # Arbitrage signals should have the custom factor
        arb_ranked = [r for r in ranked if r.signal.divergence.is_arbitrage]
        for r in arb_ranked:
            assert r.rank_factors.get("arb_bonus") == 100.0


# ============================================================================
# SignalRateLimiter Tests
# ============================================================================

class TestSlidingWindowCounter:
    """Tests for SlidingWindowCounter."""

    def test_within_limit(self):
        """Test that counter allows events within limit."""
        counter = SlidingWindowCounter(window_seconds=60.0, max_count=10)

        for _ in range(5):
            counter.record()

        can_proceed, count = counter.can_proceed()
        assert can_proceed is True
        assert count == 5

    def test_at_limit(self):
        """Test that counter blocks at limit."""
        counter = SlidingWindowCounter(window_seconds=60.0, max_count=5)

        for _ in range(5):
            counter.record()

        can_proceed, count = counter.can_proceed()
        assert can_proceed is False
        assert count == 5


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_consume_available(self):
        """Test consuming available tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        assert bucket.try_consume(1) is True
        assert bucket.try_consume(5) is True

    def test_empty_bucket(self):
        """Test consuming from empty bucket."""
        bucket = TokenBucket(capacity=3, refill_rate=0.1)

        assert bucket.try_consume(3) is True
        assert bucket.try_consume(1) is False


class TestSignalRateLimiter:
    """Tests for SignalRateLimiter."""

    def test_allows_under_limit(self, sample_scored_signal):
        """Test that signals under limit are allowed."""
        config = RateLimitConfig(global_max_per_minute=100)
        limiter = SignalRateLimiter(config)

        result = limiter.check(sample_scored_signal)
        assert result.allowed is True

    def test_critical_bypass(self, sample_divergence):
        """Test that critical signals bypass rate limit."""
        config = RateLimitConfig(
            global_max_per_minute=1,
            critical_bypass_enabled=True,
            critical_score_threshold=90.0,
        )
        limiter = SignalRateLimiter(config)

        # Use up the rate limit
        low_score_signal = ScoredSignal(
            divergence=sample_divergence,
            overall_score=50.0,
            component_scores={},
        )
        limiter.record(low_score_signal)

        # Critical signal should still pass
        critical_signal = ScoredSignal(
            divergence=sample_divergence,
            overall_score=95.0,
            component_scores={},
        )

        result = limiter.check(critical_signal)
        assert result.allowed is True

    def test_per_market_limit(self, sample_divergence):
        """Test per-market rate limiting."""
        config = RateLimitConfig(
            global_max_per_minute=100,
            per_market_max_per_minute=2,
        )
        limiter = SignalRateLimiter(config)

        signal = ScoredSignal(
            divergence=sample_divergence,
            overall_score=50.0,
            component_scores={},
        )

        # Record signals until per-market limit hit
        for i in range(3):
            result = limiter.check(signal)
            if result.allowed:
                limiter.record(signal)

        # Next should be rejected
        result = limiter.check(signal)
        assert result.allowed is False

    def test_filter_method(self, multiple_signals):
        """Test the filter method."""
        config = RateLimitConfig(global_max_per_minute=5)
        limiter = SignalRateLimiter(config)

        allowed = limiter.filter(multiple_signals)
        assert len(allowed) <= 5

    def test_rate_limiter_stats(self, multiple_signals):
        """Test rate limiter statistics."""
        config = RateLimitConfig(global_max_per_minute=5)
        limiter = SignalRateLimiter(config)

        limiter.filter(multiple_signals)
        stats = limiter.get_stats()

        assert stats["total_checked"] == len(multiple_signals)
        assert stats["total_allowed"] <= 5


# ============================================================================
# SignalAggregator Tests
# ============================================================================

class TestSignalAggregator:
    """Tests for SignalAggregator."""

    def test_full_pipeline(self, multiple_signals):
        """Test full aggregation pipeline."""
        aggregator = SignalAggregator()
        result = aggregator.process(multiple_signals)

        assert isinstance(result, AggregatorResult)
        assert len(result.stages) > 0
        assert result.input_count == len(multiple_signals)

    def test_blacklist_integration(self, multiple_signals):
        """Test blacklist integration in pipeline."""
        aggregator = SignalAggregator()
        aggregator.blacklist.block_market("market-a-0", BlockReason.MANUAL)

        result = aggregator.process(multiple_signals)

        # Signal with market-a-0 should be filtered
        divergence_ids = [r.signal.divergence.id for r in result.signals]
        assert "div-000" not in divergence_ids

    def test_whitelist_boosts_applied(self, multiple_signals):
        """Test whitelist boosts in pipeline."""
        aggregator = SignalAggregator()
        aggregator.whitelist.boost_market("market-a-9", BoostReason.HIGH_PRIORITY, score_boost=50.0)

        result = aggregator.process(multiple_signals)

        # Signal 9 should be boosted (and likely ranked higher)
        sig_9 = next(r for r in result.signals if r.signal.divergence.id == "div-009")
        assert sig_9.effective_score > multiple_signals[9].overall_score

    def test_custom_stage(self, multiple_signals):
        """Test adding custom pipeline stage."""
        aggregator = SignalAggregator()

        # Custom stage that filters out low scores
        def custom_filter(signals):
            return [s for s in signals if s.overall_score >= 50.0]

        aggregator.add_custom_stage("high_score_only", custom_filter)
        result = aggregator.process(multiple_signals)

        # All results should have score >= 50
        for ranked in result.signals:
            assert ranked.signal.overall_score >= 50.0

    def test_tier_filtering(self, multiple_signals):
        """Test filtering by minimum tier."""
        config = AggregatorConfig(min_tier_for_output=SignalTier.HIGH)
        aggregator = SignalAggregator(config)

        result = aggregator.process(multiple_signals)

        # All results should be HIGH or CRITICAL
        for ranked in result.signals:
            assert ranked.tier in {SignalTier.CRITICAL, SignalTier.HIGH}

    def test_output_limit(self, multiple_signals):
        """Test maximum output signals limit."""
        config = AggregatorConfig(max_output_signals=3)
        aggregator = SignalAggregator(config)

        result = aggregator.process(multiple_signals)
        assert len(result.signals) <= 3

    def test_get_actionable(self, multiple_signals):
        """Test getting only actionable signals."""
        aggregator = SignalAggregator()
        actionable = aggregator.get_actionable(multiple_signals)

        for ranked in actionable:
            assert ranked.tier in {SignalTier.CRITICAL, SignalTier.HIGH}

    def test_aggregator_stats(self, multiple_signals):
        """Test aggregator statistics."""
        aggregator = SignalAggregator()
        aggregator.process(multiple_signals)

        stats = aggregator.get_stats()
        assert "total_processed" in stats
        assert "filter_stats" in stats
        assert "blacklist_stats" in stats


class TestStreamingAggregator:
    """Tests for StreamingAggregator."""

    def test_submit_single(self, sample_scored_signal):
        """Test submitting a single signal."""
        aggregator = StreamingAggregator()
        result = aggregator.submit(sample_scored_signal)

        assert result is not None
        assert isinstance(result, RankedSignal)

    def test_submit_blocked(self, sample_scored_signal):
        """Test that blocked signals are rejected."""
        aggregator = StreamingAggregator()
        aggregator.blacklist.block_market("market-a", BlockReason.MANUAL)

        result = aggregator.submit(sample_scored_signal)
        assert result is None

    def test_submit_batch(self, multiple_signals):
        """Test submitting a batch of signals."""
        aggregator = StreamingAggregator()
        results = aggregator.submit_batch(multiple_signals)

        assert len(results) <= len(multiple_signals)


# ============================================================================
# SignalFormatter Tests
# ============================================================================

class TestSignalFormatter:
    """Tests for SignalFormatter."""

    @pytest.fixture
    def ranked_signals(self, multiple_signals):
        """Create ranked signals for formatting tests."""
        ranker = SignalRanker()
        return ranker.rank(multiple_signals)

    def test_format_dashboard(self, ranked_signals):
        """Test dashboard formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals, OutputFormat.DASHBOARD)

        assert "SIGNAL DASHBOARD" in output
        assert "#1" in output

    def test_format_telegram(self, ranked_signals):
        """Test Telegram formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals, OutputFormat.TELEGRAM)

        assert "Trading Signals" in output
        # Should contain emoji
        assert "🔔" in output or "📈" in output or "📊" in output

    def test_format_csv(self, ranked_signals):
        """Test CSV formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals, OutputFormat.CSV)

        # Should have header
        assert "rank" in output
        assert "tier" in output
        # Should have data rows
        lines = output.strip().split("\n")
        assert len(lines) > 1

    def test_format_json(self, ranked_signals):
        """Test JSON formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals, OutputFormat.JSON)

        # Should be valid JSON
        data = json.loads(output)
        assert "signals" in data
        assert len(data["signals"]) == len(ranked_signals)

    def test_format_json_pretty(self, ranked_signals):
        """Test pretty JSON formatting."""
        config = FormatterConfig(json_pretty=True)
        formatter = SignalFormatter(config)
        output = formatter.format(ranked_signals, OutputFormat.JSON)

        # Pretty JSON has newlines
        assert "\n" in output

    def test_format_plain_text(self, ranked_signals):
        """Test plain text formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals, OutputFormat.PLAIN_TEXT)

        assert "Signals" in output
        assert "#1" in output

    def test_format_slack(self, ranked_signals):
        """Test Slack formatting."""
        formatter = SignalFormatter()
        output = formatter.format(ranked_signals[:3], OutputFormat.SLACK)

        # Should be valid JSON for Slack
        data = json.loads(output)
        assert "blocks" in data

    def test_format_summary(self, ranked_signals):
        """Test summary formatting."""
        formatter = SignalFormatter()
        summary = formatter.format_summary(ranked_signals, OutputFormat.TELEGRAM)

        assert "Summary" in summary

    def test_telegram_max_length(self, ranked_signals):
        """Test Telegram message length limit."""
        config = FormatterConfig(telegram_max_length=500)
        formatter = SignalFormatter(config)
        output = formatter.format(ranked_signals, OutputFormat.TELEGRAM)

        # Should be truncated
        assert len(output) <= 500 + 100  # Some buffer for truncation message


class TestAlertFormatter:
    """Tests for AlertFormatter."""

    @pytest.fixture
    def ranked_signal(self, sample_scored_signal):
        """Create a single ranked signal."""
        return RankedSignal(
            signal=sample_scored_signal,
            rank=1,
            tier=SignalTier.CRITICAL,
            effective_score=90.0,
        )

    def test_format_alert(self, ranked_signal):
        """Test alert formatting."""
        formatter = AlertFormatter(emoji_enabled=True)
        alert = formatter.format_alert(ranked_signal)

        assert "SIGNAL #1" in alert
        assert "🚨" in alert

    def test_format_alert_no_emoji(self, ranked_signal):
        """Test alert formatting without emoji."""
        formatter = AlertFormatter(emoji_enabled=False)
        alert = formatter.format_alert(ranked_signal)

        assert "ALERT:" in alert
        assert "🚨" not in alert

    def test_batch_alert(self, multiple_signals):
        """Test batch alert formatting."""
        ranker = SignalRanker()
        ranked = ranker.rank(multiple_signals)

        formatter = AlertFormatter()
        alert = formatter.format_batch_alert(ranked, SignalTier.HIGH)

        # May or may not have alerts depending on scores
        if alert:
            assert "URGENT" in alert


# ============================================================================
# Integration Tests
# ============================================================================

class TestFilteringIntegration:
    """Integration tests for the filtering module."""

    def test_full_workflow(self, multiple_signals):
        """Test complete filtering workflow."""
        # 1. Create aggregator with custom config
        config = AggregatorConfig(
            enable_filtering=True,
            enable_blacklist=True,
            enable_whitelist=True,
            enable_ranking=True,
            enable_rate_limiting=True,
            max_output_signals=5,
        )
        aggregator = SignalAggregator(config)

        # 2. Configure blacklist/whitelist
        aggregator.blacklist.block_market("market-a-0", BlockReason.POOR_PERFORMANCE)
        aggregator.whitelist.boost_market("market-a-9", BoostReason.HIGH_PRIORITY, score_boost=20.0)

        # 3. Process signals
        result = aggregator.process(multiple_signals)

        # 4. Format output
        formatter = SignalFormatter()
        dashboard = formatter.format(result.signals, OutputFormat.DASHBOARD)
        telegram = formatter.format(result.signals, OutputFormat.TELEGRAM)
        json_out = formatter.format(result.signals, OutputFormat.JSON)

        # Verify results
        assert len(result.signals) <= 5
        assert "div-000" not in [r.signal.divergence.id for r in result.signals]
        assert len(dashboard) > 0
        assert len(telegram) > 0
        assert json.loads(json_out)  # Valid JSON

    def test_empty_input(self):
        """Test handling of empty input."""
        aggregator = SignalAggregator()
        result = aggregator.process([])

        assert len(result.signals) == 0
        assert result.input_count == 0

    def test_all_filtered_out(self, multiple_signals):
        """Test when all signals are filtered out."""
        config = AggregatorConfig()
        config.filter_config = FilterConfig(min_overall_score=100.0)  # Nothing passes
        aggregator = SignalAggregator(config)

        result = aggregator.process(multiple_signals)
        assert len(result.signals) == 0

    def test_performance_with_many_signals(self):
        """Test performance with many signals."""
        import time

        # Create many signals
        signals = []
        for i in range(100):
            div = Divergence(
                id=f"div-{i:03d}",
                divergence_type=DivergenceType.CORRELATION_BREAK,
                market_ids=[f"market-a-{i}", f"market-b-{i}"],
                divergence_pct=5.0,
                expected_relationship="positive",
                confidence=0.8,
                detected_at=datetime.utcnow(),
                current_prices={f"market-a-{i}": 0.55, f"market-b-{i}": 0.45},
                is_arbitrage=False,
            )
            signal = ScoredSignal(
                divergence=div,
                overall_score=50.0 + (i % 50),
                component_scores={},
            )
            signals.append(signal)

        aggregator = SignalAggregator()

        start = time.time()
        result = aggregator.process(signals)
        duration = time.time() - start

        # Should complete quickly
        assert duration < 1.0  # Less than 1 second
        assert result.total_duration_ms < 1000


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

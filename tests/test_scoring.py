"""
Comprehensive tests for the signal scoring system.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from src.signals.divergence.types import Divergence, DivergenceType, DivergenceStatus
from src.signals.scoring.types import (
    ScoredSignal,
    ComponentScore,
    ScoringConfig,
    RecommendedAction,
    Urgency,
)
from src.signals.scoring.scorer import SignalScorer
from src.signals.scoring.calibrator import ScoreCalibrator, SignalOutcome
from src.signals.scoring.recommender import (
    ActionRecommender,
    TradingRecommendation,
    PortfolioConstraints,
)
from src.signals.scoring.components.divergence import DivergenceSizeScorer
from src.signals.scoring.components.liquidity import LiquidityScorer
from src.signals.scoring.components.confidence import ConfidenceScorer
from src.signals.scoring.components.time import TimeSensitivityScorer
from src.signals.scoring.components.historical import HistoricalAccuracyScorer
from src.signals.scoring.components.risk import RiskRewardScorer
from src.models import Orderbook


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Default scoring configuration."""
    return ScoringConfig()


@pytest.fixture
def mock_orderbook():
    """Create a mock orderbook with liquidity."""
    ob = MagicMock(spec=Orderbook)
    ob.spread = 0.02
    ob.bids = [
        MagicMock(size=100),
        MagicMock(size=80),
        MagicMock(size=60),
    ]
    ob.asks = [
        MagicMock(size=90),
        MagicMock(size=70),
        MagicMock(size=50),
    ]
    return ob


@pytest.fixture
def arbitrage_divergence(mock_orderbook):
    """Create an arbitrage divergence signal."""
    return Divergence(
        id="arb-001",
        divergence_type=DivergenceType.THRESHOLD_VIOLATION,
        market_ids=["market-a", "market-b"],
        detected_at=datetime.utcnow(),
        status=DivergenceStatus.ACTIVE,
        divergence_pct=0.05,
        profit_potential=0.03,
        current_prices={"market-a": 0.45, "market-b": 0.52},
        current_orderbooks={"market-a": mock_orderbook, "market-b": mock_orderbook},
        max_executable_size=500.0,
        confidence=0.95,
        direction="BUY market-a, SELL market-b",
        is_arbitrage=True,
        supporting_evidence=["Price sum < 1.0", "High liquidity both sides"],
    )


@pytest.fixture
def directional_divergence(mock_orderbook):
    """Create a directional (non-arbitrage) divergence signal."""
    return Divergence(
        id="dir-001",
        divergence_type=DivergenceType.PRICE_SPREAD,
        market_ids=["market-a", "market-b"],
        detected_at=datetime.utcnow(),
        status=DivergenceStatus.ACTIVE,
        divergence_pct=0.08,
        profit_potential=0.04,
        current_prices={"market-a": 0.40, "market-b": 0.48},
        current_orderbooks={"market-a": mock_orderbook, "market-b": mock_orderbook},
        max_executable_size=300.0,
        confidence=0.75,
        direction="BUY market-a",
        is_arbitrage=False,
        supporting_evidence=["Statistical correlation", "Recent price movement"],
    )


@pytest.fixture
def low_liquidity_divergence():
    """Create a divergence with low liquidity."""
    return Divergence(
        id="low-liq-001",
        divergence_type=DivergenceType.CORRELATION_BREAK,
        market_ids=["market-x", "market-y"],
        detected_at=datetime.utcnow(),
        status=DivergenceStatus.ACTIVE,
        divergence_pct=0.10,
        profit_potential=0.05,
        current_prices={"market-x": 0.55, "market-y": 0.60},
        current_orderbooks={},  # No orderbook data
        max_executable_size=25.0,  # Very low liquidity
        confidence=0.50,
        direction="SELL market-y",
        is_arbitrage=False,
        supporting_evidence=[],
    )


# ============================================================================
# DivergenceSizeScorer Tests
# ============================================================================

class TestDivergenceSizeScorer:
    """Tests for divergence size scoring."""

    def test_high_divergence_scores_high(self, config, arbitrage_divergence):
        scorer = DivergenceSizeScorer(config)
        result = scorer.score(arbitrage_divergence)

        assert result.name == "divergence_size"
        assert result.score >= 40  # Score based on divergence_pct and arbitrage min
        assert result.weight == 0.25

    def test_arbitrage_minimum_score(self, config):
        """Arbitrage signals get boosted base score."""
        scorer = DivergenceSizeScorer(config)

        div = Divergence(
            id="small-arb",
            divergence_type=DivergenceType.INVERSE_SUM,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.005,  # Very small
            profit_potential=0.002,
            current_prices={"a": 0.50, "b": 0.498},
            max_executable_size=100.0,
            confidence=0.9,
            direction="BUY",
            is_arbitrage=True,
        )

        result = scorer.score(div)
        # Base score is 50 but final score includes pct adjustment
        assert result.metadata["base_score"] >= 50  # Base score minimum for arbitrage
        assert result.score > 0  # Final score is positive

    def test_very_small_divergence_scores_low(self, config):
        """Tiny divergences should score low."""
        scorer = DivergenceSizeScorer(config)

        div = Divergence(
            id="tiny",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.002,  # 0.2%
            profit_potential=0.001,
            current_prices={"a": 0.50, "b": 0.502},
            max_executable_size=100.0,
            confidence=0.5,
            direction="BUY",
            is_arbitrage=False,
        )

        result = scorer.score(div)
        assert result.score < 30


# ============================================================================
# LiquidityScorer Tests
# ============================================================================

class TestLiquidityScorer:
    """Tests for liquidity scoring."""

    def test_good_liquidity_scores_high(self, config, arbitrage_divergence):
        scorer = LiquidityScorer(config)
        result = scorer.score(arbitrage_divergence)

        assert result.name == "liquidity"
        assert result.score >= 50  # $500 should be decent

    def test_no_liquidity_scores_zero(self, config):
        """Zero liquidity should score 0."""
        scorer = LiquidityScorer(config)

        div = Divergence(
            id="no-liq",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50, "b": 0.55},
            max_executable_size=0.0,  # No liquidity
            confidence=0.7,
            direction="BUY",
            is_arbitrage=False,
        )

        result = scorer.score(div)
        assert result.score == 0

    def test_high_spread_penalty(self, config):
        """Wide spreads should reduce score."""
        scorer = LiquidityScorer(config)

        ob = MagicMock(spec=Orderbook)
        ob.spread = 0.10  # 10% spread - very wide
        ob.bids = []
        ob.asks = []

        div = Divergence(
            id="wide-spread",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50},
            current_orderbooks={"a": ob},
            max_executable_size=200.0,
            confidence=0.7,
            direction="BUY",
            is_arbitrage=False,
        )

        result = scorer.score(div)
        # Spread penalty should reduce score
        assert result.metadata["spread_penalty"] >= 30


# ============================================================================
# ConfidenceScorer Tests
# ============================================================================

class TestConfidenceScorer:
    """Tests for confidence scoring."""

    def test_high_confidence_scores_high(self, config, arbitrage_divergence):
        scorer = ConfidenceScorer(config)
        result = scorer.score(arbitrage_divergence)

        assert result.name == "confidence"
        assert result.score >= 70  # 95% confidence + arbitrage

    def test_low_confidence_scores_low(self, config, low_liquidity_divergence):
        scorer = ConfidenceScorer(config)
        result = scorer.score(low_liquidity_divergence)

        assert result.score < 60  # 50% confidence

    def test_evidence_bonus(self, config):
        """Supporting evidence should increase score."""
        scorer = ConfidenceScorer(config)

        div_no_evidence = Divergence(
            id="no-ev",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50, "b": 0.55},
            max_executable_size=200.0,
            confidence=0.70,
            direction="BUY",
            is_arbitrage=False,
            supporting_evidence=[],
        )

        div_with_evidence = Divergence(
            id="with-ev",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50, "b": 0.55},
            max_executable_size=200.0,
            confidence=0.70,
            direction="BUY",
            is_arbitrage=False,
            supporting_evidence=[
                "Multiple detection methods agree",
                "Confirmed by verification",
                "Historical precedent",
            ],
        )

        score_no_ev = scorer.score(div_no_evidence)
        score_with_ev = scorer.score(div_with_evidence)

        assert score_with_ev.score > score_no_ev.score


# ============================================================================
# TimeSensitivityScorer Tests
# ============================================================================

class TestTimeSensitivityScorer:
    """Tests for time sensitivity scoring."""

    def test_arbitrage_urgent(self, config, arbitrage_divergence):
        scorer = TimeSensitivityScorer(config)
        result = scorer.score(arbitrage_divergence)

        assert result.name == "time_sensitivity"
        # Arbitrage should be at least SOON urgency
        assert result.metadata["urgency_level"] in [Urgency.IMMEDIATE.value, Urgency.SOON.value]
        assert result.metadata["estimated_window_seconds"] <= 60

    def test_correlation_break_not_urgent(self, config, low_liquidity_divergence):
        scorer = TimeSensitivityScorer(config)
        result = scorer.score(low_liquidity_divergence)

        assert result.metadata["urgency_level"] == Urgency.WATCH.value
        assert result.metadata["estimated_window_seconds"] >= 300

    def test_expiry_increases_urgency(self, config):
        """Near expiry should increase urgency score."""
        scorer = TimeSensitivityScorer(config)

        div = Divergence(
            id="expiring",
            divergence_type=DivergenceType.PRICE_SPREAD,
            market_ids=["a", "b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50, "b": 0.55},
            max_executable_size=200.0,
            confidence=0.70,
            direction="BUY",
            is_arbitrage=False,
            expires_at=datetime.utcnow() + timedelta(seconds=30),  # 30s left
        )

        result = scorer.score(div)
        assert result.metadata["expiry_adjustment"] >= 90


# ============================================================================
# HistoricalAccuracyScorer Tests
# ============================================================================

class TestHistoricalAccuracyScorer:
    """Tests for historical accuracy scoring."""

    def test_sync_scoring_uses_defaults(self, config, arbitrage_divergence):
        scorer = HistoricalAccuracyScorer(config, db_manager=None)
        result = scorer.score_sync(arbitrage_divergence)

        assert result.name == "historical_accuracy"
        assert result.metadata["using_defaults"] is True
        # THRESHOLD_VIOLATION has 95% default accuracy
        assert result.score >= 90

    @pytest.mark.asyncio
    async def test_async_scoring_without_db(self, config, directional_divergence):
        scorer = HistoricalAccuracyScorer(config, db_manager=None)
        result = await scorer.score(directional_divergence)

        assert result.metadata["using_defaults"] is True

    def test_different_types_different_defaults(self, config):
        """Different divergence types should have different default accuracies."""
        scorer = HistoricalAccuracyScorer(config)

        threshold_div = Divergence(
            id="t1",
            divergence_type=DivergenceType.THRESHOLD_VIOLATION,
            market_ids=["a"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50},
            max_executable_size=100.0,
            confidence=0.9,
            direction="BUY",
            is_arbitrage=True,
        )

        corr_div = Divergence(
            id="c1",
            divergence_type=DivergenceType.CORRELATION_BREAK,
            market_ids=["a"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.05,
            profit_potential=0.03,
            current_prices={"a": 0.50},
            max_executable_size=100.0,
            confidence=0.5,
            direction="BUY",
            is_arbitrage=False,
        )

        t_score = scorer.score_sync(threshold_div)
        c_score = scorer.score_sync(corr_div)

        assert t_score.score > c_score.score


# ============================================================================
# RiskRewardScorer Tests
# ============================================================================

class TestRiskRewardScorer:
    """Tests for risk/reward scoring."""

    def test_arbitrage_max_score(self, config, arbitrage_divergence):
        scorer = RiskRewardScorer(config)
        result = scorer.score(arbitrage_divergence)

        assert result.name == "risk_reward"
        assert result.metadata["is_arbitrage"] is True
        assert result.metadata["kelly_fraction"] == 1.0  # Bet everything on sure thing
        assert result.score >= 80

    def test_directional_calculates_kelly(self, config, directional_divergence):
        scorer = RiskRewardScorer(config)
        result = scorer.score(directional_divergence)

        assert result.metadata["is_arbitrage"] is False
        assert "kelly_fraction" in result.metadata
        assert 0 <= result.metadata["kelly_fraction"] <= 1

    def test_position_size_calculation(self, config, directional_divergence):
        scorer = RiskRewardScorer(config)

        bankroll = 1000.0
        size = scorer.calculate_position_size(directional_divergence, bankroll)

        # Should not exceed max position fraction
        assert size <= bankroll * config.max_position_fraction
        # Should not exceed liquidity
        assert size <= directional_divergence.max_executable_size


# ============================================================================
# SignalScorer Integration Tests
# ============================================================================

class TestSignalScorer:
    """Integration tests for the main SignalScorer."""

    def test_score_arbitrage(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        result = scorer.score_divergence(arbitrage_divergence)

        assert isinstance(result, ScoredSignal)
        assert result.overall_score >= 50  # Arbitrage should score reasonably well
        assert result.recommended_action in [RecommendedAction.STRONG_BUY, RecommendedAction.BUY]
        assert result.urgency in [Urgency.IMMEDIATE, Urgency.SOON]

    def test_score_directional(self, config, directional_divergence):
        scorer = SignalScorer(config)
        result = scorer.score_divergence(directional_divergence)

        assert isinstance(result, ScoredSignal)
        assert len(result.component_scores) == 6
        assert result.score_explanation  # Should have explanations

    def test_score_low_quality(self, config, low_liquidity_divergence):
        scorer = SignalScorer(config)
        result = scorer.score_divergence(low_liquidity_divergence)

        # Low liquidity, low confidence should score poorly
        assert result.overall_score < 60
        assert result.recommended_action in [RecommendedAction.WATCH, RecommendedAction.PASS]

    def test_score_multiple_with_ranking(self, config, arbitrage_divergence, directional_divergence, low_liquidity_divergence):
        scorer = SignalScorer(config)

        divergences = [low_liquidity_divergence, directional_divergence, arbitrage_divergence]
        results = scorer.score_multiple(divergences, rank=True)

        assert len(results) == 3
        # Should be sorted by score descending
        assert results[0].overall_score >= results[1].overall_score >= results[2].overall_score
        # Ranks should be assigned
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        # Percentiles should be assigned
        assert results[0].percentile is not None

    @pytest.mark.asyncio
    async def test_async_scoring(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        result = await scorer.score_divergence_async(arbitrage_divergence)

        assert isinstance(result, ScoredSignal)
        assert result.overall_score > 0

    def test_component_scores_sum_correctly(self, config, directional_divergence):
        scorer = SignalScorer(config)
        result = scorer.score_divergence(directional_divergence)

        # Verify weighted scores contribute properly
        total_weighted = sum(cs.weighted_score for cs in result.component_scores.values())
        total_weights = sum(cs.weight for cs in result.component_scores.values())

        # Overall should be close to normalized weighted sum
        expected = (total_weighted / total_weights) if total_weights > 0 else 0
        assert abs(result.overall_score - expected) < 1.0  # Allow small rounding

    def test_score_distribution_tracking(self, config, arbitrage_divergence, directional_divergence):
        scorer = SignalScorer(config)

        # Score multiple signals to build history
        for _ in range(15):
            scorer.score_divergence(arbitrage_divergence)
            scorer.score_divergence(directional_divergence)

        dist = scorer.get_score_distribution()
        assert dist is not None
        assert dist.count >= 20
        assert dist.mean > 0
        assert dist.min_score <= dist.max_score

    def test_update_weights(self, config):
        scorer = SignalScorer(config)

        new_weights = {
            "divergence_size": 0.30,
            "liquidity": 0.20,
            "confidence": 0.15,
            "time_sensitivity": 0.15,
            "historical_accuracy": 0.10,
            "risk_reward": 0.10,
        }

        scorer.update_weights(new_weights)
        assert scorer.config.weights["divergence_size"] == 0.30

    def test_invalid_weights_rejected(self, config):
        scorer = SignalScorer(config)

        # Weights don't sum to 1.0
        with pytest.raises(ValueError):
            scorer.update_weights({
                "divergence_size": 0.50,
                "liquidity": 0.50,
                # Others not updated, sum will be wrong
            })


# ============================================================================
# ScoreCalibrator Tests
# ============================================================================

class TestScoreCalibrator:
    """Tests for the score calibrator."""

    @pytest.fixture
    def sample_outcomes(self):
        """Generate sample outcomes for testing."""
        outcomes = []
        for i in range(100):
            win = i % 3 != 0  # 67% win rate
            outcomes.append(SignalOutcome(
                divergence=MagicMock(),
                signal_time=datetime.utcnow() - timedelta(hours=i),
                score_at_detection=50 + (i % 50),
                component_scores={
                    "divergence_size": 50 + (i % 30),
                    "liquidity": 40 + (i % 40),
                    "confidence": 60 + (i % 20),
                    "time_sensitivity": 55 + (i % 25),
                    "historical_accuracy": 50 + (i % 30),
                    "risk_reward": 45 + (i % 35),
                },
                outcome="win" if win else "loss",
                profit=0.02 if win else -0.01,
                close_time=datetime.utcnow() - timedelta(hours=i-1),
                convergence_seconds=60 + i,
            ))
        return outcomes

    def test_add_outcomes(self, config, sample_outcomes):
        calibrator = ScoreCalibrator(config)

        for outcome in sample_outcomes:
            calibrator.add_outcome(outcome)

        assert len(calibrator._outcomes) == 100

    def test_backtest_threshold(self, config, sample_outcomes):
        calibrator = ScoreCalibrator(config)
        for outcome in sample_outcomes:
            calibrator.add_outcome(outcome)

        result = calibrator.backtest_threshold(60.0)

        assert result.min_score_threshold == 60.0
        assert result.total_signals > 0
        assert 0 <= result.win_rate <= 1

    def test_find_optimal_threshold(self, config, sample_outcomes):
        calibrator = ScoreCalibrator(config)
        for outcome in sample_outcomes:
            calibrator.add_outcome(outcome)

        threshold, result = calibrator.find_optimal_threshold(min_win_rate=0.5)

        assert 40 <= threshold <= 90
        assert result is not None

    def test_analyze_component_importance(self, config, sample_outcomes):
        calibrator = ScoreCalibrator(config)
        for outcome in sample_outcomes:
            calibrator.add_outcome(outcome)

        importance = calibrator.analyze_component_importance()

        assert len(importance) == 6
        for comp, metrics in importance.items():
            assert "profit_correlation" in metrics
            assert "win_correlation" in metrics
            assert "discriminative_power" in metrics

    def test_calibrate_weights_insufficient_data(self, config):
        calibrator = ScoreCalibrator(config)
        # Only add a few outcomes
        for i in range(5):
            calibrator.add_outcome(SignalOutcome(
                divergence=MagicMock(),
                signal_time=datetime.utcnow(),
                score_at_detection=60,
                component_scores={},
                outcome="win",
                profit=0.01,
                close_time=datetime.utcnow(),
                convergence_seconds=60,
            ))

        result = calibrator.calibrate_weights(min_samples=100)
        assert result is None  # Not enough data

    def test_get_weight_recommendations(self, config, sample_outcomes):
        calibrator = ScoreCalibrator(config)
        for outcome in sample_outcomes:
            calibrator.add_outcome(outcome)

        recs = calibrator.get_weight_recommendations()

        assert recs["status"] == "success"
        assert recs["sample_size"] == 100


# ============================================================================
# ActionRecommender Tests
# ============================================================================

class TestActionRecommender:
    """Tests for the action recommender."""

    def test_recommend_strong_signal(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        scored = scorer.score_divergence(arbitrage_divergence)
        rec = recommender.recommend(scored)

        assert isinstance(rec, TradingRecommendation)
        assert rec.should_execute()
        assert rec.position_size > 0
        assert rec.trade_type == "ARBITRAGE"

    def test_recommend_weak_signal(self, config, low_liquidity_divergence):
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        scored = scorer.score_divergence(low_liquidity_divergence)
        rec = recommender.recommend(scored)

        # Weak signal should not recommend execution
        assert rec.action in [RecommendedAction.WATCH, RecommendedAction.PASS]

    def test_portfolio_constraints(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        constraints = PortfolioConstraints(
            max_single_position=100.0,  # Very low limit
            max_daily_trades=2,
        )
        recommender = ActionRecommender(config, constraints=constraints, bankroll=10000.0)

        scored = scorer.score_divergence(arbitrage_divergence)
        rec = recommender.recommend(scored)

        # Position should be capped
        assert rec.position_size <= 100.0

    def test_daily_trade_limit(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        constraints = PortfolioConstraints(max_daily_trades=1)
        recommender = ActionRecommender(config, constraints=constraints, bankroll=10000.0)

        scored = scorer.score_divergence(arbitrage_divergence)

        # First trade should be allowed
        rec1 = recommender.recommend(scored)
        recommender.record_trade("market-a", 100.0)

        # Second trade should be blocked
        rec2 = recommender.recommend(scored)
        assert "Daily trade limit reached" in str(rec2.warnings)
        assert rec2.action == RecommendedAction.PASS

    def test_recommend_batch(self, config, arbitrage_divergence, directional_divergence):
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        signals = [
            scorer.score_divergence(arbitrage_divergence),
            scorer.score_divergence(directional_divergence),
        ]

        recs = recommender.recommend_batch(signals, max_recommendations=5)

        # Should return at least 1 recommendation (arbitrage)
        assert len(recs) >= 1
        # First recommendation should be the highest scored actionable signal
        if len(recs) > 1:
            assert recs[0].signal.overall_score >= recs[1].signal.overall_score

    def test_market_overlap_warning(self, config, mock_orderbook):
        """Overlapping markets should trigger warning in batch recommendations."""
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        # Two signals using same market - both need to be high-scoring to be actionable
        div1 = Divergence(
            id="d1",
            divergence_type=DivergenceType.THRESHOLD_VIOLATION,
            market_ids=["market-shared", "market-b"],
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.10,
            profit_potential=0.05,
            current_prices={"market-shared": 0.45, "market-b": 0.52},
            current_orderbooks={"market-shared": mock_orderbook, "market-b": mock_orderbook},
            max_executable_size=500.0,
            confidence=0.95,
            direction="BUY",
            is_arbitrage=True,
        )

        div2 = Divergence(
            id="d2",
            divergence_type=DivergenceType.THRESHOLD_VIOLATION,  # Also arbitrage to ensure high score
            market_ids=["market-shared", "market-c"],  # Same market!
            detected_at=datetime.utcnow(),
            status=DivergenceStatus.ACTIVE,
            divergence_pct=0.08,
            profit_potential=0.04,
            current_prices={"market-shared": 0.45, "market-c": 0.49},
            current_orderbooks={"market-shared": mock_orderbook, "market-c": mock_orderbook},
            max_executable_size=400.0,
            confidence=0.90,
            direction="BUY",
            is_arbitrage=True,
        )

        signals = [
            scorer.score_divergence(div1),
            scorer.score_divergence(div2),
        ]

        recs = recommender.recommend_batch(signals)

        # Verify we got recommendations
        assert len(recs) >= 1

        # When both signals are actionable and share a market,
        # the second one should have overlap warning or be demoted
        if len(recs) >= 2:
            # Check if either has overlap indication
            second_rec = recs[1]
            has_overlap_handling = (
                "overlap" in str(second_rec.warnings).lower() or
                second_rec.action == RecommendedAction.WATCH
            )
            assert has_overlap_handling

    def test_risk_parameters_set(self, config, directional_divergence):
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        scored = scorer.score_divergence(directional_divergence)
        rec = recommender.recommend(scored)

        assert rec.stop_loss_pct > 0
        assert rec.take_profit_pct > 0
        assert rec.max_slippage_pct > 0

    def test_to_dict_serialization(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        recommender = ActionRecommender(config, bankroll=10000.0)

        scored = scorer.score_divergence(arbitrage_divergence)
        rec = recommender.recommend(scored)

        rec_dict = rec.to_dict()

        assert "action" in rec_dict
        assert "position_size" in rec_dict
        assert "urgency" in rec_dict

    def test_status_tracking(self, config):
        constraints = PortfolioConstraints(max_daily_trades=10)
        recommender = ActionRecommender(constraints=constraints, bankroll=10000.0)

        status = recommender.get_status()
        assert status["daily_trades"] == 0
        assert status["trades_remaining"] == 10

        recommender.record_trade("market-a", 100.0, pnl=5.0)

        status = recommender.get_status()
        assert status["daily_trades"] == 1
        assert status["daily_pnl"] == 5.0

        recommender.reset_daily_stats()

        status = recommender.get_status()
        assert status["daily_trades"] == 0
        assert status["daily_pnl"] == 0.0


# ============================================================================
# ScoredSignal Tests
# ============================================================================

class TestScoredSignal:
    """Tests for ScoredSignal dataclass."""

    def test_comparison_operators(self, config, arbitrage_divergence, directional_divergence):
        scorer = SignalScorer(config)

        sig1 = scorer.score_divergence(arbitrage_divergence)
        sig2 = scorer.score_divergence(directional_divergence)

        # Comparison should work
        if sig1.overall_score > sig2.overall_score:
            assert sig1 > sig2
            assert sig2 < sig1
        else:
            assert sig2 > sig1
            assert sig1 < sig2

    def test_explain_method(self, config, arbitrage_divergence):
        scorer = SignalScorer(config)
        sig = scorer.score_divergence(arbitrage_divergence)

        explanation = sig.explain()

        assert "Signal Score" in explanation
        assert "Component Breakdown" in explanation
        assert "divergence_size" in explanation

    def test_to_dict_method(self, config, directional_divergence):
        scorer = SignalScorer(config)
        sig = scorer.score_divergence(directional_divergence)

        data = sig.to_dict()

        assert "divergence_id" in data
        assert "overall_score" in data
        assert "component_scores" in data
        assert "recommended_action" in data

    def test_get_component_breakdown(self, config, directional_divergence):
        scorer = SignalScorer(config)
        sig = scorer.score_divergence(directional_divergence)

        breakdown = sig.get_component_breakdown()

        assert len(breakdown) == 6
        for name, score in breakdown.items():
            assert 0 <= score <= 100

    def test_get_weighted_breakdown(self, config, directional_divergence):
        scorer = SignalScorer(config)
        sig = scorer.score_divergence(directional_divergence)

        weighted = sig.get_weighted_breakdown()

        assert len(weighted) == 6


# ============================================================================
# ScoringConfig Tests
# ============================================================================

class TestScoringConfig:
    """Tests for ScoringConfig."""

    def test_default_weights_sum_to_one(self):
        config = ScoringConfig()
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 0.001

    def test_validate_catches_bad_weights(self):
        config = ScoringConfig()
        config.weights["divergence_size"] = 0.5  # Now sums to > 1

        with pytest.raises(ValueError):
            config.validate()

    def test_thresholds_ordered(self):
        config = ScoringConfig()

        # Divergence thresholds should be ordered
        prev = 0
        for threshold, _, _ in config.divergence_thresholds:
            if threshold != float('inf'):
                assert threshold > prev
                prev = threshold

        # Liquidity thresholds should be ordered
        prev = 0
        for threshold, _, _ in config.liquidity_thresholds:
            if threshold != float('inf'):
                assert threshold > prev
                prev = threshold

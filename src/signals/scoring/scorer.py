"""
Main signal scorer that combines all scoring components.

Provides unified scoring interface for divergences.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.signals.divergence.types import Divergence
from src.signals.scoring.types import (
    ScoredSignal,
    ComponentScore,
    ScoringConfig,
    RecommendedAction,
    Urgency,
    ScoreDistribution,
)
from src.signals.scoring.components.divergence import DivergenceSizeScorer
from src.signals.scoring.components.liquidity import LiquidityScorer
from src.signals.scoring.components.confidence import ConfidenceScorer
from src.signals.scoring.components.time import TimeSensitivityScorer
from src.signals.scoring.components.historical import HistoricalAccuracyScorer
from src.signals.scoring.components.risk import RiskRewardScorer

logger = logging.getLogger(__name__)


class SignalScorer:
    """
    Scores divergence signals based on multiple quality factors.

    Combines component scores using configurable weights to produce
    an overall score (0-100) and trading recommendations.

    Components:
    - divergence_size: How large is the price discrepancy?
    - liquidity: Can we actually trade this profitably?
    - confidence: How reliable is this signal?
    - time_sensitivity: How urgent is this opportunity?
    - historical_accuracy: How have similar signals performed?
    - risk_reward: What's the risk-adjusted expected value?
    """

    def __init__(
        self,
        config: ScoringConfig = None,
        db_manager=None,  # DatabaseManager for historical lookups
    ):
        self.config = config or ScoringConfig()
        self.config.validate()

        self.db = db_manager

        # Initialize component scorers
        self.divergence_scorer = DivergenceSizeScorer(self.config)
        self.liquidity_scorer = LiquidityScorer(self.config)
        self.confidence_scorer = ConfidenceScorer(self.config)
        self.time_scorer = TimeSensitivityScorer(self.config)
        self.historical_scorer = HistoricalAccuracyScorer(self.config, db_manager)
        self.risk_scorer = RiskRewardScorer(self.config)

        # Score statistics for calibration
        self._score_history: List[float] = []
        self._max_history = 1000

    def score_divergence(self, divergence: Divergence) -> ScoredSignal:
        """
        Score a single divergence signal.

        This is the synchronous version that uses default historical data.
        Use score_divergence_async for full historical database lookups.

        Args:
            divergence: The divergence to score

        Returns:
            ScoredSignal with overall score and recommendations
        """
        # Calculate all component scores
        component_scores = self._calculate_component_scores_sync(divergence)

        # Calculate overall score
        overall_score = self._calculate_overall_score(component_scores)

        # Track for calibration
        self._track_score(overall_score)

        # Get time-based metadata
        time_score = component_scores.get("time_sensitivity")
        urgency = Urgency.WATCH
        window_seconds = 300

        if time_score and time_score.metadata:
            urgency_str = time_score.metadata.get("urgency_level", "WATCH")
            urgency = Urgency(urgency_str)
            window_seconds = time_score.metadata.get("estimated_window_seconds", 300)

        # Get risk-based metadata
        risk_score = component_scores.get("risk_reward")
        expected_profit = 0.0
        expected_loss = 0.0
        prob_profit = 0.5
        sharpe = 0.0

        if risk_score and risk_score.metadata:
            expected_profit = risk_score.metadata.get("profit_potential", 0.0)
            expected_loss = risk_score.metadata.get("loss_potential", 0.0)
            prob_profit = risk_score.metadata.get("win_probability", 0.5)
            sharpe = risk_score.metadata.get("sharpe_estimate", 0.0)

        # Determine recommended action
        action = self._determine_action(overall_score, urgency, divergence)

        # Calculate recommended position size
        recommended_size = self._calculate_position_size(divergence, risk_score)

        # Build explanation
        explanation = self._build_explanation(
            divergence, overall_score, component_scores, action
        )

        return ScoredSignal(
            divergence=divergence,
            scored_at=datetime.utcnow(),
            overall_score=overall_score,
            component_scores=component_scores,
            recommended_action=action,
            recommended_size=recommended_size,
            recommended_price=self._get_recommended_price(divergence),
            expected_profit=expected_profit,
            expected_loss=expected_loss,
            probability_of_profit=prob_profit,
            sharpe_estimate=sharpe,
            urgency=urgency,
            estimated_window_seconds=window_seconds,
            score_explanation=explanation,
        )

    async def score_divergence_async(self, divergence: Divergence) -> ScoredSignal:
        """
        Score a divergence with full historical database lookups.

        Use this for production scoring with historical accuracy data.
        """
        # Calculate all component scores (with async historical lookup)
        component_scores = await self._calculate_component_scores_async(divergence)

        # Calculate overall score
        overall_score = self._calculate_overall_score(component_scores)

        # Track for calibration
        self._track_score(overall_score)

        # Get time-based metadata
        time_score = component_scores.get("time_sensitivity")
        urgency = Urgency.WATCH
        window_seconds = 300

        if time_score and time_score.metadata:
            urgency_str = time_score.metadata.get("urgency_level", "WATCH")
            urgency = Urgency(urgency_str)
            window_seconds = time_score.metadata.get("estimated_window_seconds", 300)

        # Get risk-based metadata
        risk_score = component_scores.get("risk_reward")
        expected_profit = 0.0
        expected_loss = 0.0
        prob_profit = 0.5
        sharpe = 0.0

        if risk_score and risk_score.metadata:
            expected_profit = risk_score.metadata.get("profit_potential", 0.0)
            expected_loss = risk_score.metadata.get("loss_potential", 0.0)
            prob_profit = risk_score.metadata.get("win_probability", 0.5)
            sharpe = risk_score.metadata.get("sharpe_estimate", 0.0)

        # Determine recommended action
        action = self._determine_action(overall_score, urgency, divergence)

        # Calculate recommended position size
        recommended_size = self._calculate_position_size(divergence, risk_score)

        # Build explanation
        explanation = self._build_explanation(
            divergence, overall_score, component_scores, action
        )

        return ScoredSignal(
            divergence=divergence,
            scored_at=datetime.utcnow(),
            overall_score=overall_score,
            component_scores=component_scores,
            recommended_action=action,
            recommended_size=recommended_size,
            recommended_price=self._get_recommended_price(divergence),
            expected_profit=expected_profit,
            expected_loss=expected_loss,
            probability_of_profit=prob_profit,
            sharpe_estimate=sharpe,
            urgency=urgency,
            estimated_window_seconds=window_seconds,
            score_explanation=explanation,
        )

    def score_multiple(
        self,
        divergences: List[Divergence],
        rank: bool = True
    ) -> List[ScoredSignal]:
        """
        Score multiple divergences and optionally rank them.

        Args:
            divergences: List of divergences to score
            rank: Whether to assign ranks based on overall score

        Returns:
            List of ScoredSignals, sorted by score descending if rank=True
        """
        scored = [self.score_divergence(d) for d in divergences]

        if rank:
            scored = self._rank_signals(scored)

        return scored

    async def score_multiple_async(
        self,
        divergences: List[Divergence],
        rank: bool = True
    ) -> List[ScoredSignal]:
        """
        Score multiple divergences asynchronously with ranking.
        """
        import asyncio

        # Score all in parallel
        scored = await asyncio.gather(
            *[self.score_divergence_async(d) for d in divergences]
        )
        scored = list(scored)

        if rank:
            scored = self._rank_signals(scored)

        return scored

    def _calculate_component_scores_sync(
        self,
        divergence: Divergence
    ) -> Dict[str, ComponentScore]:
        """Calculate all component scores synchronously."""
        scores = {}

        # Divergence size
        div_score = self.divergence_scorer.score(divergence)
        div_score.weighted_score = div_score.score * div_score.weight
        scores["divergence_size"] = div_score

        # Liquidity
        liq_score = self.liquidity_scorer.score(divergence)
        liq_score.weighted_score = liq_score.score * liq_score.weight
        scores["liquidity"] = liq_score

        # Confidence
        conf_score = self.confidence_scorer.score(divergence)
        conf_score.weighted_score = conf_score.score * conf_score.weight
        scores["confidence"] = conf_score

        # Time sensitivity
        time_score = self.time_scorer.score(divergence)
        time_score.weighted_score = time_score.score * time_score.weight
        scores["time_sensitivity"] = time_score

        # Historical accuracy (sync version)
        hist_score = self.historical_scorer.score_sync(divergence)
        hist_score.weighted_score = hist_score.score * hist_score.weight
        scores["historical_accuracy"] = hist_score

        # Risk/reward
        risk_score = self.risk_scorer.score(divergence)
        risk_score.weighted_score = risk_score.score * risk_score.weight
        scores["risk_reward"] = risk_score

        return scores

    async def _calculate_component_scores_async(
        self,
        divergence: Divergence
    ) -> Dict[str, ComponentScore]:
        """Calculate all component scores with async historical lookup."""
        scores = {}

        # Divergence size
        div_score = self.divergence_scorer.score(divergence)
        div_score.weighted_score = div_score.score * div_score.weight
        scores["divergence_size"] = div_score

        # Liquidity
        liq_score = self.liquidity_scorer.score(divergence)
        liq_score.weighted_score = liq_score.score * liq_score.weight
        scores["liquidity"] = liq_score

        # Confidence
        conf_score = self.confidence_scorer.score(divergence)
        conf_score.weighted_score = conf_score.score * conf_score.weight
        scores["confidence"] = conf_score

        # Time sensitivity
        time_score = self.time_scorer.score(divergence)
        time_score.weighted_score = time_score.score * time_score.weight
        scores["time_sensitivity"] = time_score

        # Historical accuracy (async version with DB lookup)
        hist_score = await self.historical_scorer.score(divergence)
        hist_score.weighted_score = hist_score.score * hist_score.weight
        scores["historical_accuracy"] = hist_score

        # Risk/reward
        risk_score = self.risk_scorer.score(divergence)
        risk_score.weighted_score = risk_score.score * risk_score.weight
        scores["risk_reward"] = risk_score

        return scores

    def _calculate_overall_score(
        self,
        component_scores: Dict[str, ComponentScore]
    ) -> float:
        """Calculate weighted overall score from components."""
        total = sum(cs.weighted_score for cs in component_scores.values())

        # Normalize by total weight (should be 1.0 but be safe)
        total_weight = sum(cs.weight for cs in component_scores.values())
        if total_weight > 0:
            total = total / total_weight * 100 / 100  # Convert back to 0-100 scale

        return max(0.0, min(100.0, total))

    def _determine_action(
        self,
        score: float,
        urgency: Urgency,
        divergence: Divergence
    ) -> RecommendedAction:
        """Determine recommended trading action."""
        thresholds = self.config.action_thresholds

        # For arbitrage, lower the threshold
        is_arbitrage = divergence.is_arbitrage

        if is_arbitrage:
            if score >= thresholds["BUY"]:
                return RecommendedAction.STRONG_BUY
            elif score >= thresholds["WATCH"]:
                return RecommendedAction.BUY

        # Normal threshold logic
        if score >= thresholds["STRONG_BUY"]:
            return RecommendedAction.STRONG_BUY
        elif score >= thresholds["BUY"]:
            return RecommendedAction.BUY
        elif score >= thresholds["WATCH"]:
            return RecommendedAction.WATCH
        else:
            return RecommendedAction.PASS

    def _calculate_position_size(
        self,
        divergence: Divergence,
        risk_score: Optional[ComponentScore]
    ) -> float:
        """Calculate recommended position size."""
        # Use Kelly-based sizing from risk scorer if available
        if risk_score and risk_score.metadata:
            kelly = risk_score.metadata.get("kelly_fraction", 0.0)
            if kelly > 0:
                # Apply to a nominal $1000 bankroll
                # In production, this would use actual bankroll
                nominal_bankroll = 1000.0
                kelly_size = nominal_bankroll * kelly

                # Cap at max position and available liquidity
                max_size = nominal_bankroll * self.config.max_position_fraction
                return min(kelly_size, max_size, divergence.max_executable_size)

        # Default: smaller of 2% of nominal bankroll or liquidity
        nominal_bankroll = 1000.0
        return min(
            nominal_bankroll * 0.02,
            divergence.max_executable_size
        )

    def _get_recommended_price(self, divergence: Divergence) -> float:
        """Get recommended entry price based on divergence type."""
        # For arbitrage, use exact prices needed
        if divergence.is_arbitrage:
            prices = list(divergence.current_prices.values())
            if len(prices) >= 2:
                # Return the price we'd pay (buy the cheaper)
                return min(prices)

        # For directional, use current best price
        prices = list(divergence.current_prices.values())
        if prices:
            return min(prices)  # Conservative: lowest price

        return 0.0

    def _build_explanation(
        self,
        divergence: Divergence,
        score: float,
        component_scores: Dict[str, ComponentScore],
        action: RecommendedAction
    ) -> List[str]:
        """Build list of explanation points."""
        explanations = []

        # Top factors (sorted by weighted score)
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )

        # Include top 3 factors
        for name, cs in sorted_components[:3]:
            if cs.weighted_score > 0:
                explanations.append(cs.explanation)

        # Add action-specific context
        if action == RecommendedAction.STRONG_BUY:
            if divergence.is_arbitrage:
                explanations.append("Arbitrage opportunity with high confidence")
            else:
                explanations.append("Strong signal across multiple factors")
        elif action == RecommendedAction.BUY:
            explanations.append("Good opportunity worth pursuing")
        elif action == RecommendedAction.WATCH:
            explanations.append("Monitor for better entry conditions")
        else:
            # Find the weakest factor
            weakest = sorted_components[-1]
            explanations.append(f"Weak {weakest[0]} score limits opportunity")

        return explanations

    def _rank_signals(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """Rank signals by overall score and assign percentiles."""
        if not signals:
            return signals

        # Sort by score descending
        signals.sort(key=lambda x: x.overall_score, reverse=True)

        # Assign ranks
        for i, signal in enumerate(signals):
            signal.rank = i + 1
            signal.percentile = (len(signals) - i) / len(signals) * 100

        return signals

    def _track_score(self, score: float) -> None:
        """Track score for calibration statistics."""
        self._score_history.append(score)
        if len(self._score_history) > self._max_history:
            self._score_history = self._score_history[-self._max_history:]

    def get_score_distribution(self) -> Optional[ScoreDistribution]:
        """Get statistics about recent score distribution."""
        if len(self._score_history) < 10:
            return None

        import statistics

        scores = self._score_history
        sorted_scores = sorted(scores)
        n = len(scores)

        # Calculate percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            idx = int(n * p / 100)
            percentiles[p] = sorted_scores[min(idx, n-1)]

        # Count by action threshold
        action_counts = {
            "STRONG_BUY": sum(1 for s in scores if s >= self.config.action_thresholds["STRONG_BUY"]),
            "BUY": sum(1 for s in scores if self.config.action_thresholds["BUY"] <= s < self.config.action_thresholds["STRONG_BUY"]),
            "WATCH": sum(1 for s in scores if self.config.action_thresholds["WATCH"] <= s < self.config.action_thresholds["BUY"]),
            "PASS": sum(1 for s in scores if s < self.config.action_thresholds["WATCH"]),
        }

        return ScoreDistribution(
            count=n,
            mean=statistics.mean(scores),
            median=statistics.median(scores),
            std=statistics.stdev(scores) if n > 1 else 0,
            min_score=min(scores),
            max_score=max(scores),
            percentiles=percentiles,
            action_counts=action_counts,
        )

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update component weights.

        Args:
            new_weights: Dict mapping component name to new weight

        Raises:
            ValueError if weights don't sum to 1.0
        """
        for name, weight in new_weights.items():
            if name in self.config.weights:
                self.config.weights[name] = weight

        self.config.validate()

        # Update in individual scorers
        for name, weight in self.config.weights.items():
            # Each scorer will use config.weights in their score() method
            pass

        logger.info(f"Updated scoring weights: {self.config.weights}")

    def get_component_scores(self, divergence: Divergence) -> Dict[str, float]:
        """Get individual component scores (0-100) without full scoring."""
        scores = self._calculate_component_scores_sync(divergence)
        return {name: cs.score for name, cs in scores.items()}

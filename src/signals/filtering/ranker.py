"""
Signal ranking and categorization.

Provides sophisticated ranking to prioritize signals for action.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, Set
from enum import Enum
from collections import defaultdict

from src.signals.scoring.types import ScoredSignal, RecommendedAction, Urgency

logger = logging.getLogger(__name__)


class SignalTier(str, Enum):
    """Signal priority tier."""
    CRITICAL = "critical"  # Act immediately
    HIGH = "high"  # Act within minutes
    MEDIUM = "medium"  # Act within hour
    LOW = "low"  # Monitor
    IGNORE = "ignore"  # Skip


@dataclass
class RankingConfig:
    """Configuration for signal ranking."""
    # Tier thresholds (score-based)
    critical_score_threshold: float = 85.0
    high_score_threshold: float = 70.0
    medium_score_threshold: float = 50.0
    low_score_threshold: float = 30.0

    # Urgency multipliers
    urgent_multiplier: float = 1.5
    time_sensitive_multiplier: float = 1.3
    normal_multiplier: float = 1.0

    # Arbitrage boost
    arbitrage_boost: float = 20.0

    # Diversification
    max_signals_per_market: int = 3
    diversification_penalty: float = 5.0  # Per duplicate market

    # Recency boost (newer signals get priority)
    recency_boost_max: float = 10.0
    recency_window_seconds: float = 60.0


@dataclass
class RankedSignal:
    """A signal with ranking information."""
    signal: ScoredSignal
    rank: int
    tier: SignalTier
    effective_score: float
    rank_factors: Dict[str, float] = field(default_factory=dict)


class SignalRanker:
    """
    Ranks signals by actionability.

    Combines multiple factors:
    - Base score from SignalScorer
    - Urgency multiplier
    - Arbitrage boost
    - Recency boost
    - Diversification penalty
    - Whitelist boosts
    """

    def __init__(self, config: RankingConfig = None):
        self.config = config or RankingConfig()

        # Statistics
        self._total_ranked = 0
        self._tier_distribution: Dict[SignalTier, int] = defaultdict(int)

    def rank(
        self,
        signals: List[ScoredSignal],
        whitelist_boosts: Optional[Dict[str, float]] = None
    ) -> List[RankedSignal]:
        """
        Rank signals by effective score.

        Returns signals sorted by rank (1 = highest priority).
        """
        if not signals:
            return []

        whitelist_boosts = whitelist_boosts or {}

        # Calculate effective scores
        scored_signals = []
        market_counts: Dict[str, int] = defaultdict(int)

        for signal in signals:
            effective_score, factors = self._calculate_effective_score(
                signal,
                market_counts,
                whitelist_boosts,
            )

            # Update market counts for diversification
            for market_id in signal.divergence.market_ids:
                market_counts[market_id] += 1

            tier = self._determine_tier(effective_score)

            scored_signals.append({
                "signal": signal,
                "effective_score": effective_score,
                "tier": tier,
                "factors": factors,
            })

        # Sort by effective score descending
        scored_signals.sort(key=lambda x: x["effective_score"], reverse=True)

        # Create ranked results
        results = []
        for rank, item in enumerate(scored_signals, start=1):
            ranked = RankedSignal(
                signal=item["signal"],
                rank=rank,
                tier=item["tier"],
                effective_score=item["effective_score"],
                rank_factors=item["factors"],
            )
            results.append(ranked)

            # Track statistics
            self._total_ranked += 1
            self._tier_distribution[ranked.tier] += 1

        return results

    def _calculate_effective_score(
        self,
        signal: ScoredSignal,
        market_counts: Dict[str, int],
        whitelist_boosts: Dict[str, float],
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate effective score for ranking.

        Returns (effective_score, factors_dict).
        """
        factors = {}

        # Start with base score
        base_score = signal.overall_score
        factors["base_score"] = base_score

        effective_score = base_score

        # Urgency multiplier
        urgency_mult = self._get_urgency_multiplier(signal)
        factors["urgency_multiplier"] = urgency_mult
        effective_score *= urgency_mult

        # Arbitrage boost
        if signal.divergence.is_arbitrage:
            effective_score += self.config.arbitrage_boost
            factors["arbitrage_boost"] = self.config.arbitrage_boost

        # Recency boost
        recency_boost = self._calculate_recency_boost(signal)
        if recency_boost > 0:
            effective_score += recency_boost
            factors["recency_boost"] = recency_boost

        # Whitelist boost
        for market_id in signal.divergence.market_ids:
            if market_id in whitelist_boosts:
                boost = whitelist_boosts[market_id]
                effective_score += boost
                factors[f"whitelist_boost_{market_id}"] = boost

        # Diversification penalty
        diversification_penalty = self._calculate_diversification_penalty(
            signal, market_counts
        )
        if diversification_penalty > 0:
            effective_score -= diversification_penalty
            factors["diversification_penalty"] = -diversification_penalty

        # Confidence adjustment
        if hasattr(signal, 'confidence') and signal.confidence is not None:
            confidence_factor = signal.confidence / 100.0
            adjustment = (confidence_factor - 0.5) * 10  # -5 to +5
            effective_score += adjustment
            factors["confidence_adjustment"] = adjustment

        factors["effective_score"] = effective_score
        return effective_score, factors

    def _get_urgency_multiplier(self, signal: ScoredSignal) -> float:
        """Get multiplier based on signal urgency."""
        if hasattr(signal, 'urgency'):
            if signal.urgency == Urgency.IMMEDIATE:
                return self.config.urgent_multiplier
            elif signal.urgency == Urgency.SOON:
                return self.config.time_sensitive_multiplier

        # Infer urgency from recommended action
        if hasattr(signal, 'recommendation') and signal.recommendation:
            if signal.recommendation.urgency == Urgency.IMMEDIATE:
                return self.config.urgent_multiplier
            elif signal.recommendation.urgency == Urgency.SOON:
                return self.config.time_sensitive_multiplier

        return self.config.normal_multiplier

    def _calculate_recency_boost(self, signal: ScoredSignal) -> float:
        """Calculate boost for recent signals."""
        age_seconds = (datetime.utcnow() - signal.scored_at).total_seconds()

        if age_seconds > self.config.recency_window_seconds:
            return 0.0

        # Linear decay from max boost to 0
        recency_factor = 1 - (age_seconds / self.config.recency_window_seconds)
        return self.config.recency_boost_max * recency_factor

    def _calculate_diversification_penalty(
        self,
        signal: ScoredSignal,
        market_counts: Dict[str, int]
    ) -> float:
        """Calculate penalty for over-concentration in markets."""
        total_penalty = 0.0

        for market_id in signal.divergence.market_ids:
            count = market_counts.get(market_id, 0)
            if count >= self.config.max_signals_per_market:
                # Apply increasing penalty for each duplicate
                excess = count - self.config.max_signals_per_market + 1
                total_penalty += self.config.diversification_penalty * excess

        return total_penalty

    def _determine_tier(self, effective_score: float) -> SignalTier:
        """Determine signal tier based on effective score."""
        if effective_score >= self.config.critical_score_threshold:
            return SignalTier.CRITICAL
        elif effective_score >= self.config.high_score_threshold:
            return SignalTier.HIGH
        elif effective_score >= self.config.medium_score_threshold:
            return SignalTier.MEDIUM
        elif effective_score >= self.config.low_score_threshold:
            return SignalTier.LOW
        else:
            return SignalTier.IGNORE

    def get_top_n(
        self,
        signals: List[ScoredSignal],
        n: int,
        min_tier: SignalTier = SignalTier.LOW
    ) -> List[RankedSignal]:
        """
        Get top N signals above a minimum tier.
        """
        ranked = self.rank(signals)

        # Filter by tier
        tier_order = [
            SignalTier.CRITICAL,
            SignalTier.HIGH,
            SignalTier.MEDIUM,
            SignalTier.LOW,
            SignalTier.IGNORE,
        ]
        min_tier_idx = tier_order.index(min_tier)
        allowed_tiers = set(tier_order[:min_tier_idx + 1])

        filtered = [r for r in ranked if r.tier in allowed_tiers]

        return filtered[:n]

    def get_by_tier(
        self,
        signals: List[ScoredSignal]
    ) -> Dict[SignalTier, List[RankedSignal]]:
        """
        Group ranked signals by tier.
        """
        ranked = self.rank(signals)

        by_tier: Dict[SignalTier, List[RankedSignal]] = defaultdict(list)
        for r in ranked:
            by_tier[r.tier].append(r)

        return dict(by_tier)

    def get_diversified(
        self,
        signals: List[ScoredSignal],
        max_per_market: int = 2,
        total_limit: int = 10
    ) -> List[RankedSignal]:
        """
        Get diversified set of top signals.

        Limits exposure to any single market.
        """
        ranked = self.rank(signals)

        market_counts: Dict[str, int] = defaultdict(int)
        diversified = []

        for r in ranked:
            if len(diversified) >= total_limit:
                break

            # Check market limits
            can_add = True
            for market_id in r.signal.divergence.market_ids:
                if market_counts[market_id] >= max_per_market:
                    can_add = False
                    break

            if can_add:
                diversified.append(r)
                for market_id in r.signal.divergence.market_ids:
                    market_counts[market_id] += 1

        return diversified

    def get_actionable(
        self,
        signals: List[ScoredSignal],
        allowed_actions: Optional[Set[RecommendedAction]] = None
    ) -> List[RankedSignal]:
        """
        Get signals with specific recommended actions.
        """
        if allowed_actions is None:
            allowed_actions = {
                RecommendedAction.STRONG_BUY,
                RecommendedAction.BUY,
            }

        ranked = self.rank(signals)

        return [
            r for r in ranked
            if hasattr(r.signal, 'recommendation')
            and r.signal.recommendation
            and r.signal.recommendation.action in allowed_actions
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get ranking statistics."""
        return {
            "total_ranked": self._total_ranked,
            "tier_distribution": {
                tier.value: count
                for tier, count in self._tier_distribution.items()
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_ranked = 0
        self._tier_distribution.clear()


class MultiFactorRanker:
    """
    Advanced ranker with customizable factor weights.

    Allows fine-tuning the ranking formula.
    """

    def __init__(self):
        self.factor_weights: Dict[str, float] = {
            "score": 1.0,
            "divergence_pct": 0.5,
            "liquidity": 0.3,
            "urgency": 0.4,
            "confidence": 0.2,
            "recency": 0.2,
        }
        self.custom_factors: List[Callable[[ScoredSignal], float]] = []

    def set_weight(self, factor: str, weight: float) -> None:
        """Set weight for a factor."""
        self.factor_weights[factor] = weight

    def add_custom_factor(
        self,
        name: str,
        factor_func: Callable[[ScoredSignal], float],
        weight: float = 1.0
    ) -> None:
        """Add a custom ranking factor."""
        self.factor_weights[name] = weight
        self.custom_factors.append((name, factor_func))

    def rank(self, signals: List[ScoredSignal]) -> List[RankedSignal]:
        """Rank signals using weighted factors."""
        scored = []

        for signal in signals:
            factors = {}
            weighted_sum = 0.0
            total_weight = 0.0

            # Score factor
            if "score" in self.factor_weights:
                weight = self.factor_weights["score"]
                factors["score"] = signal.overall_score
                weighted_sum += signal.overall_score * weight
                total_weight += weight

            # Divergence factor
            if "divergence_pct" in self.factor_weights:
                weight = self.factor_weights["divergence_pct"]
                div_score = min(signal.divergence.divergence_pct * 10, 100)
                factors["divergence_pct"] = div_score
                weighted_sum += div_score * weight
                total_weight += weight

            # Liquidity factor
            if "liquidity" in self.factor_weights:
                weight = self.factor_weights["liquidity"]
                liq_component = signal.component_scores.get("liquidity")
                if liq_component:
                    factors["liquidity"] = liq_component.score
                    weighted_sum += liq_component.score * weight
                    total_weight += weight

            # Confidence factor
            if "confidence" in self.factor_weights and hasattr(signal, 'confidence'):
                weight = self.factor_weights["confidence"]
                if signal.confidence is not None:
                    factors["confidence"] = signal.confidence
                    weighted_sum += signal.confidence * weight
                    total_weight += weight

            # Custom factors
            for name, func in self.custom_factors:
                try:
                    value = func(signal)
                    weight = self.factor_weights.get(name, 1.0)
                    factors[name] = value
                    weighted_sum += value * weight
                    total_weight += weight
                except Exception as e:
                    logger.warning(f"Custom factor {name} error: {e}")

            effective_score = weighted_sum / total_weight if total_weight > 0 else 0

            scored.append({
                "signal": signal,
                "effective_score": effective_score,
                "factors": factors,
            })

        # Sort by effective score
        scored.sort(key=lambda x: x["effective_score"], reverse=True)

        # Create ranked results
        results = []
        for rank, item in enumerate(scored, start=1):
            # Determine tier based on effective score
            if item["effective_score"] >= 85:
                tier = SignalTier.CRITICAL
            elif item["effective_score"] >= 70:
                tier = SignalTier.HIGH
            elif item["effective_score"] >= 50:
                tier = SignalTier.MEDIUM
            elif item["effective_score"] >= 30:
                tier = SignalTier.LOW
            else:
                tier = SignalTier.IGNORE

            results.append(RankedSignal(
                signal=item["signal"],
                rank=rank,
                tier=tier,
                effective_score=item["effective_score"],
                rank_factors=item["factors"],
            ))

        return results

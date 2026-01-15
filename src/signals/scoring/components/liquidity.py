"""
Liquidity scoring component.

Scores signals based on market liquidity and tradability.
"""
from typing import Optional

from src.signals.divergence.types import Divergence
from src.signals.scoring.types import ComponentScore, ScoringConfig
from src.models import Orderbook


class LiquidityScorer:
    """
    Scores divergences based on available liquidity.

    Considers:
    - Executable size (how much can be traded)
    - Bid-ask spread (transaction costs)
    - Orderbook depth (sustainability)
    """

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
        self.thresholds = self.config.liquidity_thresholds

    def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on liquidity assessment.

        Higher score = more tradeable with lower friction.
        """
        executable_size = divergence.max_executable_size
        orderbooks = divergence.current_orderbooks

        # Base score from executable size
        size_score = self._calculate_size_score(executable_size)

        # Spread penalty
        spread_penalty = self._calculate_spread_penalty(orderbooks)

        # Depth bonus
        depth_bonus = self._calculate_depth_bonus(orderbooks)

        # Combine
        final_score = size_score - spread_penalty + depth_bonus
        final_score = max(0, min(100, final_score))

        explanation = self._build_explanation(
            executable_size, size_score, spread_penalty, depth_bonus
        )

        return ComponentScore(
            name="liquidity",
            score=final_score,
            weight=self.config.weights.get("liquidity", 0.20),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "executable_size": executable_size,
                "size_score": size_score,
                "spread_penalty": spread_penalty,
                "depth_bonus": depth_bonus,
            }
        )

    def _calculate_size_score(self, size: float) -> float:
        """Calculate score based on executable size."""
        if size <= 0:
            return 0

        prev_threshold = 0.0

        for threshold, min_score, max_score in self.thresholds:
            if size <= threshold:
                if threshold == float('inf'):
                    return max_score

                range_size = threshold - prev_threshold
                if range_size > 0:
                    position = (size - prev_threshold) / range_size
                    return min_score + position * (max_score - min_score)
                return min_score

            prev_threshold = threshold

        return 100.0

    def _calculate_spread_penalty(
        self,
        orderbooks: dict[str, Orderbook]
    ) -> float:
        """
        Calculate penalty based on bid-ask spreads.

        Wide spreads = high transaction costs = penalty.
        """
        if not orderbooks:
            return 10  # Default penalty for missing data

        spreads = []
        for ob in orderbooks.values():
            if ob and ob.spread is not None:
                spreads.append(ob.spread)

        if not spreads:
            return 10

        avg_spread = sum(spreads) / len(spreads)

        # Penalty scale:
        # 1% spread = 5 point penalty
        # 2% spread = 15 point penalty
        # 5% spread = 30 point penalty
        # >10% spread = 50 point penalty
        if avg_spread < 0.01:
            return avg_spread * 500  # 0-5
        elif avg_spread < 0.02:
            return 5 + (avg_spread - 0.01) * 1000  # 5-15
        elif avg_spread < 0.05:
            return 15 + (avg_spread - 0.02) * 500  # 15-30
        elif avg_spread < 0.10:
            return 30 + (avg_spread - 0.05) * 400  # 30-50
        else:
            return 50

    def _calculate_depth_bonus(
        self,
        orderbooks: dict[str, Orderbook]
    ) -> float:
        """
        Calculate bonus for deep orderbooks.

        Deep books = more sustainable opportunity = bonus.
        """
        if not orderbooks:
            return 0

        total_depth = 0
        for ob in orderbooks.values():
            if ob:
                # Sum up top 3 levels on each side
                bid_depth = sum(
                    float(b.size) for b in (ob.bids or [])[:3]
                )
                ask_depth = sum(
                    float(a.size) for a in (ob.asks or [])[:3]
                )
                total_depth += bid_depth + ask_depth

        # Bonus scale:
        # $500 total depth = 2 points
        # $1000 = 5 points
        # $5000 = 10 points
        # $10000+ = 15 points
        if total_depth < 500:
            return total_depth / 250  # 0-2
        elif total_depth < 1000:
            return 2 + (total_depth - 500) / 166  # 2-5
        elif total_depth < 5000:
            return 5 + (total_depth - 1000) / 800  # 5-10
        else:
            return min(15, 10 + (total_depth - 5000) / 1000)

    def _build_explanation(
        self,
        size: float,
        size_score: float,
        spread_penalty: float,
        depth_bonus: float
    ) -> str:
        """Build human-readable explanation."""
        if size <= 0:
            return "No executable liquidity available"

        if size < 50:
            quality = "Very low"
        elif size < 200:
            quality = "Low"
        elif size < 500:
            quality = "Moderate"
        elif size < 1000:
            quality = "Good"
        else:
            quality = "Excellent"

        parts = [f"{quality} liquidity: ${size:.0f} executable"]

        if spread_penalty > 10:
            parts.append(f"wide spreads (-{spread_penalty:.0f})")
        if depth_bonus > 5:
            parts.append(f"deep books (+{depth_bonus:.0f})")

        return ". ".join(parts)

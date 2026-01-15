"""
Divergence size scoring component.

Scores signals based on the magnitude of price divergence.
"""
from typing import List, Tuple

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import ComponentScore, ScoringConfig


class DivergenceSizeScorer:
    """
    Scores divergences based on the size of the price gap.

    Larger divergences generally represent better opportunities,
    but with diminishing returns at very large sizes (may indicate
    fundamental differences rather than temporary mispricings).
    """

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
        self.thresholds = self.config.divergence_thresholds
        self.arbitrage_min = self.config.arbitrage_min_score

    def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on divergence magnitude.

        For arbitrage (guaranteed profit): minimum score of 50
        For directional: scaled based on size thresholds
        """
        div_amount = divergence.divergence_amount
        is_arbitrage = divergence.is_arbitrage

        # Calculate base score from magnitude
        base_score = self._calculate_size_score(div_amount)

        # Boost arbitrage opportunities
        if is_arbitrage:
            base_score = max(base_score, self.arbitrage_min)
            # Scale remaining range
            if div_amount > 0:
                # Additional boost for larger arb profits
                arb_bonus = min(div_amount * 500, 50)  # Up to +50 for 10¢
                base_score = min(100, base_score + arb_bonus)

        # Adjust for divergence percentage (relative size matters)
        pct_score = self._calculate_percentage_adjustment(divergence)
        final_score = base_score * 0.7 + pct_score * 0.3

        # Build explanation
        explanation = self._build_explanation(divergence, base_score, is_arbitrage)

        return ComponentScore(
            name="divergence_size",
            score=min(100, max(0, final_score)),
            weight=self.config.weights.get("divergence_size", 0.25),
            weighted_score=0.0,  # Will be set by scorer
            explanation=explanation,
            metadata={
                "divergence_amount": div_amount,
                "divergence_pct": divergence.divergence_pct,
                "is_arbitrage": is_arbitrage,
                "base_score": base_score,
            }
        )

    def _calculate_size_score(self, amount: float) -> float:
        """Calculate score based on absolute divergence size."""
        prev_threshold = 0.0
        prev_max = 0

        for threshold, min_score, max_score in self.thresholds:
            if amount <= threshold:
                # Linear interpolation within range
                if threshold == float('inf'):
                    # For the last bucket, scale logarithmically
                    # but cap at max_score
                    return max_score

                range_size = threshold - prev_threshold
                if range_size > 0:
                    position = (amount - prev_threshold) / range_size
                    return min_score + position * (max_score - min_score)
                return min_score

            prev_threshold = threshold
            prev_max = max_score

        return 100.0  # Should not reach here

    def _calculate_percentage_adjustment(self, divergence: Divergence) -> float:
        """
        Adjust score based on percentage divergence.

        A 5¢ divergence on a 50¢ market (10%) is more significant
        than a 5¢ divergence on a 95¢ market (5.3%).
        """
        pct = divergence.divergence_pct

        if pct <= 0:
            return 0

        # Score based on percentage
        # 1% = 20, 5% = 50, 10% = 70, 20% = 90, >30% = 100
        if pct < 0.01:
            return pct * 2000  # 0-20
        elif pct < 0.05:
            return 20 + (pct - 0.01) * 750  # 20-50
        elif pct < 0.10:
            return 50 + (pct - 0.05) * 400  # 50-70
        elif pct < 0.20:
            return 70 + (pct - 0.10) * 200  # 70-90
        else:
            return min(100, 90 + (pct - 0.20) * 50)  # 90-100

    def _build_explanation(
        self,
        divergence: Divergence,
        base_score: float,
        is_arbitrage: bool
    ) -> str:
        """Build human-readable explanation."""
        amount = divergence.divergence_amount
        pct = divergence.divergence_pct

        if is_arbitrage:
            return (
                f"Arbitrage opportunity: {amount*100:.1f}¢ guaranteed profit "
                f"({pct:.1%} spread). Minimum score: {self.arbitrage_min}"
            )

        if amount < 0.02:
            quality = "Small"
        elif amount < 0.04:
            quality = "Moderate"
        elif amount < 0.06:
            quality = "Good"
        else:
            quality = "Large"

        return (
            f"{quality} divergence: {amount*100:.1f}¢ ({pct:.1%}). "
            f"Base score: {base_score:.0f}"
        )

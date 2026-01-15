"""
Confidence scoring component.

Scores signals based on reliability and confidence factors.
"""
from typing import Optional, List

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import ComponentScore, ScoringConfig


class ConfidenceScorer:
    """
    Scores divergences based on signal confidence.

    Considers:
    - Correlation confidence (from Phase 2)
    - Multiple detection methods agreeing
    - Manual verification status
    - Market liquidity (liquid = reliable prices)
    - Divergence type reliability
    """

    # Reliability by divergence type
    TYPE_RELIABILITY = {
        DivergenceType.THRESHOLD_VIOLATION: 1.0,    # Logical, certain
        DivergenceType.INVERSE_SUM: 0.95,           # Mathematical
        DivergenceType.PRICE_SPREAD: 0.85,          # Usually reliable
        DivergenceType.LEAD_LAG_OPPORTUNITY: 0.70,  # Statistical
        DivergenceType.LAGGING_MARKET: 0.65,        # Uncertain timing
        DivergenceType.CORRELATION_BREAK: 0.50,     # May be fundamental
    }

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()

    def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on confidence assessment.

        Higher score = more reliable signal.
        """
        # Start with base confidence from divergence
        base_confidence = divergence.confidence

        # Adjust for divergence type reliability
        type_reliability = self.TYPE_RELIABILITY.get(
            divergence.divergence_type, 0.7
        )

        # Adjust for supporting evidence
        evidence_bonus = self._calculate_evidence_bonus(divergence)

        # Adjust for arbitrage status
        arbitrage_bonus = 0.1 if divergence.is_arbitrage else 0.0

        # Liquidity adjustment (illiquid = less reliable prices)
        liquidity_adjustment = self._calculate_liquidity_adjustment(divergence)

        # Combine factors
        combined_confidence = (
            base_confidence * 0.4 +
            type_reliability * 0.3 +
            evidence_bonus * 0.15 +
            arbitrage_bonus * 0.05 +
            liquidity_adjustment * 0.1
        )

        # Convert to 0-100 score
        final_score = combined_confidence * 100

        explanation = self._build_explanation(
            divergence, base_confidence, type_reliability, evidence_bonus
        )

        return ComponentScore(
            name="confidence",
            score=max(0, min(100, final_score)),
            weight=self.config.weights.get("confidence", 0.20),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "base_confidence": base_confidence,
                "type_reliability": type_reliability,
                "evidence_bonus": evidence_bonus,
                "arbitrage_bonus": arbitrage_bonus,
                "liquidity_adjustment": liquidity_adjustment,
            }
        )

    def _calculate_evidence_bonus(self, divergence: Divergence) -> float:
        """
        Calculate bonus based on supporting evidence.

        More evidence = higher confidence.
        """
        evidence = divergence.supporting_evidence or []
        num_evidence = len(evidence)

        # Check for specific high-value evidence
        has_multiple_methods = any(
            "method" in e.lower() or "detection" in e.lower()
            for e in evidence
        )
        has_verification = any(
            "verif" in e.lower() or "confirm" in e.lower()
            for e in evidence
        )

        base_bonus = min(num_evidence * 0.1, 0.3)  # Up to 0.3 for evidence count

        if has_multiple_methods:
            base_bonus += 0.1
        if has_verification:
            base_bonus += 0.1

        return min(0.5, base_bonus)

    def _calculate_liquidity_adjustment(self, divergence: Divergence) -> float:
        """
        Adjust confidence based on liquidity.

        Liquid markets have more reliable prices.
        """
        size = divergence.max_executable_size

        if size <= 0:
            return 0.3  # Penalty for no liquidity
        elif size < 100:
            return 0.5
        elif size < 500:
            return 0.7
        elif size < 1000:
            return 0.85
        else:
            return 1.0

    def _build_explanation(
        self,
        divergence: Divergence,
        base_confidence: float,
        type_reliability: float,
        evidence_bonus: float
    ) -> str:
        """Build human-readable explanation."""
        parts = []

        # Base confidence level
        if base_confidence >= 0.9:
            parts.append(f"High base confidence ({base_confidence:.0%})")
        elif base_confidence >= 0.7:
            parts.append(f"Good base confidence ({base_confidence:.0%})")
        else:
            parts.append(f"Moderate confidence ({base_confidence:.0%})")

        # Type reliability
        dtype = divergence.divergence_type
        if type_reliability >= 0.9:
            parts.append(f"{dtype.value} is highly reliable")
        elif type_reliability < 0.6:
            parts.append(f"{dtype.value} has inherent uncertainty")

        # Arbitrage
        if divergence.is_arbitrage:
            parts.append("guaranteed profit (arbitrage)")

        # Evidence
        if evidence_bonus > 0.2:
            parts.append(f"{len(divergence.supporting_evidence)} supporting factors")

        return ". ".join(parts).capitalize()

"""
Time sensitivity scoring component.

Scores signals based on urgency and time constraints.
"""
from datetime import datetime, timedelta
from typing import Optional

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import ComponentScore, ScoringConfig, Urgency


class TimeSensitivityScorer:
    """
    Scores divergences based on time sensitivity and urgency.

    Higher score = more urgent (needs immediate action).

    Considers:
    - Divergence type (arbitrage = immediate)
    - Expected time window
    - Recent price volatility
    - Time to expiration
    """

    # Base urgency by divergence type
    TYPE_URGENCY = {
        DivergenceType.THRESHOLD_VIOLATION: 95,      # Pure arbitrage, act NOW
        DivergenceType.INVERSE_SUM: 90,              # Arbitrage-like
        DivergenceType.LEAD_LAG_OPPORTUNITY: 85,     # Short window
        DivergenceType.PRICE_SPREAD: 60,             # May persist
        DivergenceType.LAGGING_MARKET: 50,           # Often takes time
        DivergenceType.CORRELATION_BREAK: 30,        # May be permanent
    }

    # Estimated window by type (seconds)
    TYPE_WINDOWS = {
        DivergenceType.THRESHOLD_VIOLATION: 30,
        DivergenceType.INVERSE_SUM: 60,
        DivergenceType.LEAD_LAG_OPPORTUNITY: 60,
        DivergenceType.PRICE_SPREAD: 300,
        DivergenceType.LAGGING_MARKET: 600,
        DivergenceType.CORRELATION_BREAK: 3600,
    }

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()

    def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on time sensitivity.

        Higher score = more urgent, needs immediate action.
        """
        dtype = divergence.divergence_type

        # Base urgency from type
        base_urgency = self.TYPE_URGENCY.get(dtype, 50)

        # Adjust for arbitrage
        if divergence.is_arbitrage:
            base_urgency = max(base_urgency, 90)

        # Adjust for profit potential (larger = more urgent)
        profit_adjustment = self._calculate_profit_urgency(divergence)

        # Adjust for expiration
        expiry_adjustment = self._calculate_expiry_adjustment(divergence)

        # Volatility adjustment (high volatility = more urgent)
        volatility_adjustment = self._calculate_volatility_adjustment(divergence)

        # Combine
        final_score = (
            base_urgency * 0.5 +
            profit_adjustment * 0.2 +
            expiry_adjustment * 0.15 +
            volatility_adjustment * 0.15
        )

        # Determine urgency level and estimated window
        urgency, window = self._determine_urgency(divergence, final_score)

        explanation = self._build_explanation(
            divergence, base_urgency, urgency
        )

        return ComponentScore(
            name="time_sensitivity",
            score=max(0, min(100, final_score)),
            weight=self.config.weights.get("time_sensitivity", 0.15),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "base_urgency": base_urgency,
                "profit_adjustment": profit_adjustment,
                "expiry_adjustment": expiry_adjustment,
                "volatility_adjustment": volatility_adjustment,
                "urgency_level": urgency.value,
                "estimated_window_seconds": window,
            }
        )

    def _calculate_profit_urgency(self, divergence: Divergence) -> float:
        """
        Higher potential profit = more urgent.

        Others will be racing for it too.
        """
        profit = divergence.profit_potential

        if profit <= 0:
            return 0
        elif profit < 0.02:
            return profit * 2000  # 0-40
        elif profit < 0.05:
            return 40 + (profit - 0.02) * 1000  # 40-70
        elif profit < 0.10:
            return 70 + (profit - 0.05) * 400  # 70-90
        else:
            return 90 + min(10, (profit - 0.10) * 100)  # 90-100

    def _calculate_expiry_adjustment(self, divergence: Divergence) -> float:
        """
        Closer to expiry = more urgent.
        """
        if divergence.expires_at is None:
            return 50  # Default

        now = datetime.utcnow()
        time_left = (divergence.expires_at - now).total_seconds()

        if time_left <= 0:
            return 100  # Already expired!
        elif time_left < 60:
            return 95
        elif time_left < 300:
            return 80
        elif time_left < 600:
            return 60
        elif time_left < 1800:
            return 40
        else:
            return 20

    def _calculate_volatility_adjustment(self, divergence: Divergence) -> float:
        """
        Higher volatility = more urgent (opportunity may disappear).

        Note: This uses divergence percentage as a proxy for volatility.
        In production, would use actual market volatility data.
        """
        # Use divergence percentage as volatility proxy
        div_pct = divergence.divergence_pct

        if div_pct < 0.02:
            return 30  # Low volatility indication
        elif div_pct < 0.05:
            return 50
        elif div_pct < 0.10:
            return 70
        else:
            return 90  # High volatility

    def _determine_urgency(
        self,
        divergence: Divergence,
        score: float
    ) -> tuple[Urgency, int]:
        """
        Determine urgency level and estimated window.
        """
        dtype = divergence.divergence_type
        base_window = self.TYPE_WINDOWS.get(dtype, 300)

        # Adjust window based on score
        if score >= 85:
            urgency = Urgency.IMMEDIATE
            window = max(10, base_window // 10)  # Much shorter
        elif score >= 60:
            urgency = Urgency.SOON
            window = max(30, base_window // 2)
        else:
            urgency = Urgency.WATCH
            window = base_window

        return urgency, window

    def _build_explanation(
        self,
        divergence: Divergence,
        base_urgency: float,
        urgency: Urgency
    ) -> str:
        """Build human-readable explanation."""
        dtype = divergence.divergence_type

        parts = []

        if urgency == Urgency.IMMEDIATE:
            parts.append("URGENT: Act immediately")
        elif urgency == Urgency.SOON:
            parts.append("Time-sensitive: Trade soon")
        else:
            parts.append("Can wait: Monitor situation")

        # Type-specific context
        if divergence.is_arbitrage:
            parts.append("arbitrage may close quickly")
        elif dtype == DivergenceType.LEAD_LAG_OPPORTUNITY:
            lag = divergence.metadata.get("lag_seconds", 60)
            parts.append(f"~{lag}s lag window")
        elif dtype == DivergenceType.CORRELATION_BREAK:
            parts.append("may be permanent change")

        return ". ".join(parts).capitalize()

    def get_urgency_for_signal(self, divergence: Divergence) -> tuple[Urgency, int]:
        """
        Get urgency level and window without full scoring.

        Convenience method for quick urgency checks.
        """
        dtype = divergence.divergence_type
        base_urgency = self.TYPE_URGENCY.get(dtype, 50)

        if divergence.is_arbitrage:
            base_urgency = max(base_urgency, 90)

        return self._determine_urgency(divergence, base_urgency)

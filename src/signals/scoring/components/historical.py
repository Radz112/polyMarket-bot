"""
Historical accuracy scoring component.

Scores signals based on past performance of similar signals.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import ComponentScore, ScoringConfig

logger = logging.getLogger(__name__)


class HistoricalAccuracyScorer:
    """
    Scores divergences based on historical performance.

    Looks at past signals with:
    - Same divergence type
    - Same/similar market pairs
    - Similar market conditions

    Calculates win rate and average profit.
    """

    # Default accuracy by type (used when no history available)
    DEFAULT_ACCURACY = {
        DivergenceType.THRESHOLD_VIOLATION: 0.95,   # Almost always works
        DivergenceType.INVERSE_SUM: 0.90,           # Very reliable
        DivergenceType.PRICE_SPREAD: 0.70,          # Usually converges
        DivergenceType.LEAD_LAG_OPPORTUNITY: 0.65,  # Statistical
        DivergenceType.LAGGING_MARKET: 0.60,        # Often works
        DivergenceType.CORRELATION_BREAK: 0.45,     # Uncertain
    }

    def __init__(
        self,
        config: ScoringConfig = None,
        db_manager=None,  # DatabaseManager for historical lookups
    ):
        self.config = config or ScoringConfig()
        self.db = db_manager
        self.lookback_days = self.config.historical_lookback_days

        # Cache for historical stats
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)

    async def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on historical accuracy of similar signals.

        Uses database lookups when available, falls back to defaults.
        """
        dtype = divergence.divergence_type

        # Try to get historical stats
        history = await self._get_historical_stats(divergence)

        if history:
            win_rate = history.get("win_rate", 0.5)
            avg_profit = history.get("avg_profit", 0.0)
            sample_size = history.get("sample_size", 0)
            avg_time = history.get("avg_convergence_seconds", 300)
        else:
            # Use defaults
            win_rate = self.DEFAULT_ACCURACY.get(dtype, 0.5)
            avg_profit = 0.0
            sample_size = 0
            avg_time = 300

        # Calculate score
        # Win rate is primary factor
        win_rate_score = win_rate * 100

        # Bonus for profitable history
        profit_bonus = min(20, avg_profit * 500) if avg_profit > 0 else 0

        # Confidence adjustment based on sample size
        if sample_size < 5:
            confidence_mult = 0.6  # Low confidence
        elif sample_size < 20:
            confidence_mult = 0.8
        elif sample_size < 50:
            confidence_mult = 0.9
        else:
            confidence_mult = 1.0

        final_score = (win_rate_score + profit_bonus) * confidence_mult

        explanation = self._build_explanation(
            dtype, win_rate, avg_profit, sample_size
        )

        return ComponentScore(
            name="historical_accuracy",
            score=max(0, min(100, final_score)),
            weight=self.config.weights.get("historical_accuracy", 0.10),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "sample_size": sample_size,
                "avg_convergence_seconds": avg_time,
                "using_defaults": sample_size == 0,
            }
        )

    def score_sync(self, divergence: Divergence) -> ComponentScore:
        """
        Synchronous version using defaults only.

        Use when database is not available or for quick scoring.
        """
        dtype = divergence.divergence_type
        win_rate = self.DEFAULT_ACCURACY.get(dtype, 0.5)

        # Arbitrage boost
        if divergence.is_arbitrage:
            win_rate = max(win_rate, 0.95)

        final_score = win_rate * 100

        explanation = self._build_explanation(dtype, win_rate, 0.0, 0)

        return ComponentScore(
            name="historical_accuracy",
            score=max(0, min(100, final_score)),
            weight=self.config.weights.get("historical_accuracy", 0.10),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "win_rate": win_rate,
                "avg_profit": 0.0,
                "sample_size": 0,
                "using_defaults": True,
            }
        )

    async def _get_historical_stats(
        self,
        divergence: Divergence
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical performance stats for similar signals.

        Returns dict with win_rate, avg_profit, sample_size, etc.
        """
        if self.db is None:
            return None

        # Create cache key
        dtype = divergence.divergence_type
        markets = tuple(sorted(divergence.market_ids))
        cache_key = f"{dtype.value}:{markets}"

        # Check cache
        if cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key, datetime.min)
            if datetime.utcnow() - cache_time < self._cache_ttl:
                return self._cache[cache_key]

        try:
            stats = await self._query_historical_stats(divergence)
            if stats:
                self._cache[cache_key] = stats
                self._cache_time[cache_key] = datetime.utcnow()
            return stats
        except Exception as e:
            logger.warning(f"Failed to get historical stats: {e}")
            return None

    async def _query_historical_stats(
        self,
        divergence: Divergence
    ) -> Optional[Dict[str, Any]]:
        """
        Query database for historical signal performance.

        This would need a signals_history table that tracks:
        - Signal type
        - Market IDs
        - Detected at
        - Outcome (win/loss)
        - Profit/loss
        - Convergence time

        For now, returns None (no history available).
        In production, implement actual DB query.
        """
        # TODO: Implement when signals_history table is available
        # Example query structure:
        #
        # SELECT
        #     COUNT(*) as sample_size,
        #     AVG(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as win_rate,
        #     AVG(profit) as avg_profit,
        #     AVG(convergence_seconds) as avg_convergence_seconds
        # FROM signals_history
        # WHERE
        #     divergence_type = ?
        #     AND detected_at > NOW() - INTERVAL ? DAY
        #     AND (market_a = ? AND market_b = ? OR market_a = ? AND market_b = ?)
        # GROUP BY divergence_type

        return None

    def _build_explanation(
        self,
        dtype: DivergenceType,
        win_rate: float,
        avg_profit: float,
        sample_size: int
    ) -> str:
        """Build human-readable explanation."""
        if sample_size == 0:
            return (
                f"No history available for {dtype.value}. "
                f"Using default win rate: {win_rate:.0%}"
            )

        quality = "Strong" if win_rate >= 0.7 else "Moderate" if win_rate >= 0.5 else "Weak"

        parts = [
            f"{quality} historical performance: {win_rate:.0%} win rate"
        ]

        if avg_profit != 0:
            parts.append(f"avg profit {avg_profit*100:.1f}¢")

        parts.append(f"based on {sample_size} signals")

        return ". ".join(parts)

    async def record_outcome(
        self,
        divergence: Divergence,
        outcome: str,  # "win" or "loss"
        profit: float,
        convergence_seconds: int
    ) -> None:
        """
        Record signal outcome for future analysis.

        Call this after a signal is resolved.
        """
        if self.db is None:
            return

        # TODO: Implement when signals_history table is available
        # INSERT INTO signals_history (
        #     divergence_type, market_ids, detected_at,
        #     outcome, profit, convergence_seconds
        # ) VALUES (?, ?, ?, ?, ?, ?)

        # Invalidate cache
        markets = tuple(sorted(divergence.market_ids))
        cache_key = f"{divergence.divergence_type.value}:{markets}"
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._cache_time[cache_key]

        logger.info(
            f"Recorded outcome for {divergence.divergence_type.value}: "
            f"{outcome}, profit={profit:.4f}"
        )

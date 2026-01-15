"""
Signal filtering system.

Provides configurable filters to remove noise and focus on actionable signals.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Set, Any
from enum import Enum

from src.signals.scoring.types import ScoredSignal, RecommendedAction, Urgency
from src.signals.divergence.types import DivergenceType

logger = logging.getLogger(__name__)


class FilterResult(str, Enum):
    """Result of a filter check."""
    PASS = "pass"
    REJECT = "reject"
    BOOST = "boost"


@dataclass
class FilterDecision:
    """Detailed decision from a filter."""
    result: FilterResult
    filter_name: str
    reason: str
    score_adjustment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Configuration for signal filters."""
    # Score thresholds
    min_overall_score: float = 30.0
    high_score_threshold: float = 70.0

    # Liquidity thresholds
    min_liquidity_score: float = 20.0
    min_combined_liquidity: float = 1000.0  # $1000 minimum

    # Divergence thresholds
    min_divergence_pct: float = 2.0  # 2% minimum
    max_divergence_pct: float = 50.0  # Likely data error above this

    # Staleness
    max_signal_age_seconds: int = 300  # 5 minutes
    max_price_age_seconds: int = 60  # 1 minute

    # Correlation strength
    min_correlation_strength: float = 0.5

    # Market conditions
    min_market_volume_24h: float = 10000.0  # $10k daily volume
    max_spread_pct: float = 5.0  # 5% max spread

    # Historical performance
    min_historical_win_rate: float = 0.0  # Disabled by default
    min_historical_trades: int = 0  # Disabled by default

    # Divergence types to accept
    allowed_divergence_types: Set[DivergenceType] = field(
        default_factory=lambda: {
            DivergenceType.CORRELATION_BREAK,
            DivergenceType.PRICE_SPREAD,
            DivergenceType.LAGGING_MARKET,
            DivergenceType.LEAD_LAG_OPPORTUNITY,
        }
    )


class SignalFilter:
    """
    Configurable signal filter.

    Applies multiple filter criteria to signals and can either:
    - Reject signals that don't meet criteria
    - Boost signals that exceed thresholds
    - Pass signals through unchanged
    """

    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self._custom_filters: List[Callable[[ScoredSignal], FilterDecision]] = []

        # Statistics
        self._total_processed = 0
        self._total_passed = 0
        self._total_rejected = 0
        self._rejection_reasons: Dict[str, int] = {}

    def filter(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """
        Filter a list of signals.

        Returns signals that pass all filters.
        """
        passed = []
        for signal in signals:
            decisions = self.evaluate(signal)
            if self._should_pass(decisions):
                passed.append(signal)
        return passed

    def filter_with_decisions(
        self,
        signals: List[ScoredSignal]
    ) -> List[tuple[ScoredSignal, List[FilterDecision]]]:
        """
        Filter signals and return decisions for each.

        Returns list of (signal, decisions) tuples for signals that pass.
        """
        results = []
        for signal in signals:
            decisions = self.evaluate(signal)
            if self._should_pass(decisions):
                results.append((signal, decisions))
        return results

    def evaluate(self, signal: ScoredSignal) -> List[FilterDecision]:
        """
        Evaluate a signal against all filters.

        Returns list of filter decisions.
        """
        self._total_processed += 1
        decisions = []

        # Built-in filters
        decisions.append(self._filter_by_score(signal))
        decisions.append(self._filter_by_liquidity(signal))
        decisions.append(self._filter_by_divergence(signal))
        decisions.append(self._filter_by_staleness(signal))
        decisions.append(self._filter_by_correlation(signal))
        decisions.append(self._filter_by_market_conditions(signal))
        decisions.append(self._filter_by_divergence_type(signal))

        # Custom filters
        for custom_filter in self._custom_filters:
            try:
                decisions.append(custom_filter(signal))
            except Exception as e:
                logger.warning(f"Custom filter error: {e}")

        # Track statistics
        if self._should_pass(decisions):
            self._total_passed += 1
        else:
            self._total_rejected += 1
            for d in decisions:
                if d.result == FilterResult.REJECT:
                    self._rejection_reasons[d.filter_name] = (
                        self._rejection_reasons.get(d.filter_name, 0) + 1
                    )

        return decisions

    def _should_pass(self, decisions: List[FilterDecision]) -> bool:
        """Check if signal should pass based on decisions."""
        return not any(d.result == FilterResult.REJECT for d in decisions)

    def _filter_by_score(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by overall score threshold."""
        if signal.overall_score < self.config.min_overall_score:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="score",
                reason=f"Score {signal.overall_score:.1f} below minimum {self.config.min_overall_score}",
            )

        if signal.overall_score >= self.config.high_score_threshold:
            return FilterDecision(
                result=FilterResult.BOOST,
                filter_name="score",
                reason=f"High score {signal.overall_score:.1f}",
                score_adjustment=5.0,
            )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="score",
            reason="Score within acceptable range",
        )

    def _filter_by_liquidity(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by liquidity requirements."""
        liquidity_score = signal.component_scores.get("liquidity", None)

        if liquidity_score and liquidity_score.score < self.config.min_liquidity_score:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="liquidity",
                reason=f"Liquidity score {liquidity_score.score:.1f} below minimum",
            )

        # Check raw liquidity if available
        div = signal.divergence
        if hasattr(div, 'combined_liquidity') and div.combined_liquidity is not None:
            if div.combined_liquidity < self.config.min_combined_liquidity:
                return FilterDecision(
                    result=FilterResult.REJECT,
                    filter_name="liquidity",
                    reason=f"Combined liquidity ${div.combined_liquidity:.0f} below minimum",
                )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="liquidity",
            reason="Liquidity acceptable",
        )

    def _filter_by_divergence(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by divergence magnitude."""
        div_pct = signal.divergence.divergence_pct

        if div_pct < self.config.min_divergence_pct:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="divergence",
                reason=f"Divergence {div_pct:.1f}% below minimum {self.config.min_divergence_pct}%",
            )

        if div_pct > self.config.max_divergence_pct:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="divergence",
                reason=f"Divergence {div_pct:.1f}% above maximum (likely data error)",
            )

        # Boost for strong divergence
        if div_pct >= 10.0:
            return FilterDecision(
                result=FilterResult.BOOST,
                filter_name="divergence",
                reason=f"Strong divergence {div_pct:.1f}%",
                score_adjustment=3.0,
            )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="divergence",
            reason="Divergence within acceptable range",
        )

    def _filter_by_staleness(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by signal and price staleness."""
        now = datetime.utcnow()

        # Check signal age
        signal_age = (now - signal.scored_at).total_seconds()
        if signal_age > self.config.max_signal_age_seconds:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="staleness",
                reason=f"Signal age {signal_age:.0f}s exceeds maximum {self.config.max_signal_age_seconds}s",
            )

        # Check price staleness if available
        div = signal.divergence
        if hasattr(div, 'price_timestamps') and div.price_timestamps:
            oldest_price = min(div.price_timestamps.values())
            price_age = (now - oldest_price).total_seconds()
            if price_age > self.config.max_price_age_seconds:
                return FilterDecision(
                    result=FilterResult.REJECT,
                    filter_name="staleness",
                    reason=f"Price data {price_age:.0f}s old",
                )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="staleness",
            reason="Data is fresh",
        )

    def _filter_by_correlation(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by correlation strength."""
        div = signal.divergence

        if hasattr(div, 'correlation_strength') and div.correlation_strength is not None:
            if div.correlation_strength < self.config.min_correlation_strength:
                return FilterDecision(
                    result=FilterResult.REJECT,
                    filter_name="correlation",
                    reason=f"Correlation strength {div.correlation_strength:.2f} below minimum",
                )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="correlation",
            reason="Correlation acceptable",
        )

    def _filter_by_market_conditions(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by market conditions (volume, spread)."""
        div = signal.divergence

        # Check 24h volume if available
        if hasattr(div, 'volume_24h') and div.volume_24h is not None:
            if div.volume_24h < self.config.min_market_volume_24h:
                return FilterDecision(
                    result=FilterResult.REJECT,
                    filter_name="market_conditions",
                    reason=f"24h volume ${div.volume_24h:.0f} below minimum",
                )

        # Check spread if available
        if hasattr(div, 'spread_pct') and div.spread_pct is not None:
            if div.spread_pct > self.config.max_spread_pct:
                return FilterDecision(
                    result=FilterResult.REJECT,
                    filter_name="market_conditions",
                    reason=f"Spread {div.spread_pct:.1f}% exceeds maximum",
                )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="market_conditions",
            reason="Market conditions acceptable",
        )

    def _filter_by_divergence_type(self, signal: ScoredSignal) -> FilterDecision:
        """Filter by divergence type."""
        div_type = signal.divergence.divergence_type

        if div_type not in self.config.allowed_divergence_types:
            return FilterDecision(
                result=FilterResult.REJECT,
                filter_name="divergence_type",
                reason=f"Divergence type {div_type.value} not allowed",
            )

        # Boost for arbitrage opportunities
        if signal.divergence.is_arbitrage:
            return FilterDecision(
                result=FilterResult.BOOST,
                filter_name="divergence_type",
                reason="Arbitrage opportunity detected",
                score_adjustment=10.0,
            )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="divergence_type",
            reason="Divergence type accepted",
        )

    def filter_by_historical_performance(
        self,
        signal: ScoredSignal,
        performance_data: Dict[str, Any]
    ) -> FilterDecision:
        """
        Filter by historical performance of similar signals.

        This is called separately as it requires external performance data.
        """
        market_ids = frozenset(signal.divergence.market_ids)
        market_key = "_".join(sorted(market_ids))

        if market_key in performance_data:
            perf = performance_data[market_key]
            win_rate = perf.get("win_rate", 0)
            total_trades = perf.get("total_trades", 0)

            if total_trades >= self.config.min_historical_trades:
                if win_rate < self.config.min_historical_win_rate:
                    return FilterDecision(
                        result=FilterResult.REJECT,
                        filter_name="historical",
                        reason=f"Historical win rate {win_rate:.1%} below minimum",
                    )

                if win_rate >= 0.7:
                    return FilterDecision(
                        result=FilterResult.BOOST,
                        filter_name="historical",
                        reason=f"Strong historical performance ({win_rate:.1%} win rate)",
                        score_adjustment=5.0,
                    )

        return FilterDecision(
            result=FilterResult.PASS,
            filter_name="historical",
            reason="Historical data not available or not required",
        )

    def add_custom_filter(
        self,
        filter_func: Callable[[ScoredSignal], FilterDecision]
    ) -> None:
        """Add a custom filter function."""
        self._custom_filters.append(filter_func)

    def remove_custom_filters(self) -> None:
        """Remove all custom filters."""
        self._custom_filters.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        return {
            "total_processed": self._total_processed,
            "total_passed": self._total_passed,
            "total_rejected": self._total_rejected,
            "pass_rate": (
                self._total_passed / self._total_processed
                if self._total_processed > 0 else 0
            ),
            "rejection_reasons": dict(self._rejection_reasons),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_processed = 0
        self._total_passed = 0
        self._total_rejected = 0
        self._rejection_reasons.clear()


class CompositeFilter:
    """
    Combines multiple filters with AND/OR logic.
    """

    def __init__(self, require_all: bool = True):
        """
        Initialize composite filter.

        Args:
            require_all: If True, all filters must pass. If False, any filter passing is enough.
        """
        self.require_all = require_all
        self.filters: List[SignalFilter] = []

    def add_filter(self, filter_: SignalFilter) -> None:
        """Add a filter to the composite."""
        self.filters.append(filter_)

    def filter(self, signals: List[ScoredSignal]) -> List[ScoredSignal]:
        """Apply composite filter logic."""
        if not self.filters:
            return signals

        if self.require_all:
            # AND logic: signal must pass all filters
            result = signals
            for f in self.filters:
                result = f.filter(result)
            return result
        else:
            # OR logic: signal passes if it passes any filter
            passed_ids = set()
            passed_signals = []

            for f in self.filters:
                for signal in f.filter(signals):
                    # Use divergence.id as unique identifier
                    sig_id = signal.divergence.id
                    if sig_id not in passed_ids:
                        passed_ids.add(sig_id)
                        passed_signals.append(signal)

            return passed_signals

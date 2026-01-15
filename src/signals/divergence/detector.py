"""
Divergence detection for correlated markets.

Detects when correlated markets deviate from their expected relationships,
identifying potential arbitrage and trading opportunities.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from src.models import MarketCorrelation, CorrelationType, Orderbook
from src.correlation.logical.rules import LogicalRule, LogicalRuleType
from src.correlation.store import CorrelationStore
from src.database.redis_cache import CacheManager
from src.config.settings import Config

from .types import Divergence, DivergenceType, DivergenceStatus, DivergenceConfig
from .price_monitor import PriceMonitor
from .liquidity import LiquidityAssessor, TwoSidedLiquidity

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """
    Detects divergences between correlated markets.

    Monitors price relationships and identifies opportunities when markets
    deviate from expected correlations.
    """

    def __init__(
        self,
        price_monitor: PriceMonitor,
        correlation_store: Optional[CorrelationStore] = None,
        cache: Optional[CacheManager] = None,
        config: Optional[DivergenceConfig] = None,
    ):
        self.monitor = price_monitor
        self.correlation_store = correlation_store
        self.cache = cache
        self.config = config or DivergenceConfig()
        self.liquidity = LiquidityAssessor()

        # Active divergences tracking
        self._active_divergences: Dict[str, Divergence] = {}

        # Debouncing: track last detection time per market pair
        self._last_detection: Dict[str, float] = {}

        # Statistics
        self._detection_count = 0
        self._false_positive_count = 0

    # --- Main Detection Entry Points ---

    async def detect_all_divergences(
        self,
        correlations: Optional[List[MarketCorrelation]] = None,
        rules: Optional[List[LogicalRule]] = None,
    ) -> List[Divergence]:
        """
        Run all divergence detectors and return combined results.

        Args:
            correlations: List of market correlations to check (if None, fetches from store)
            rules: List of logical rules to check (if None, uses empty list)

        Returns:
            List of detected divergences
        """
        # Fetch correlations from store if not provided
        if correlations is None and self.correlation_store:
            correlations = await self.correlation_store.get_all_correlations()
        correlations = correlations or []
        rules = rules or []

        all_divergences: List[Divergence] = []

        # Run all detection methods concurrently
        detection_tasks = [
            self.detect_price_spread_divergence(correlations),
            self.detect_threshold_violations(rules),
            self.detect_inverse_sum_divergence(rules),
            self.detect_lagging_market(correlations),
            self.detect_correlation_break(correlations),
            self.detect_lead_lag_opportunity(correlations),
        ]

        results = await asyncio.gather(*detection_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Detection error: {result}")
            elif isinstance(result, list):
                all_divergences.extend(result)

        # Deduplicate and filter
        all_divergences = self._deduplicate_divergences(all_divergences)
        all_divergences = self._apply_debouncing(all_divergences)

        # Update active divergences
        for div in all_divergences:
            self._active_divergences[div.id] = div
            self._detection_count += 1

        # Clean up expired divergences
        self._cleanup_expired()

        return all_divergences

    # --- Synchronous Detection (for testing/simple use) ---

    def detect_all(
        self,
        correlations: List[MarketCorrelation],
        rules: List[LogicalRule],
    ) -> List[Divergence]:
        """
        Synchronous version of detect_all_divergences for backward compatibility.
        """
        divergences = []

        # Filter correlations by confidence
        high_conf_corrs = [
            c for c in correlations
            if c.confidence >= self.config.min_correlation_confidence
        ]

        divergences.extend(self._check_spreads_sync(high_conf_corrs))
        divergences.extend(self._check_rule_violations_sync(rules))
        divergences.extend(self._check_lead_lag_sync(high_conf_corrs))

        return divergences

    # --- Individual Detection Methods ---

    async def detect_price_spread_divergence(
        self,
        correlations: List[MarketCorrelation],
    ) -> List[Divergence]:
        """
        Detect price spread divergences for equivalent/correlated markets.

        For markets that should have the same price, flags when prices differ
        by more than the threshold.

        Example:
        - "Trump wins" on market A at 54¢
        - "Trump wins" on market B at 58¢
        - Spread = 4¢, flag divergence
        """
        results = []
        threshold = self.config.get_threshold(DivergenceType.PRICE_SPREAD)

        # Filter to high-confidence correlations
        candidates = [
            c for c in correlations
            if c.confidence >= self.config.min_correlation_confidence
            and c.correlation_type in (CorrelationType.EQUIVALENT, CorrelationType.MATHEMATICAL)
        ]

        for corr in candidates:
            try:
                pa = self.monitor.get_current_price(corr.market_a_id)
                pb = self.monitor.get_current_price(corr.market_b_id)

                if pa is None or pb is None:
                    continue

                spread = abs(pa - pb)

                if spread > threshold:
                    # Determine direction
                    if pa < pb:
                        direction = "BUY_A_SELL_B"
                        cheap_market = corr.market_a_id
                        expensive_market = corr.market_b_id
                    else:
                        direction = "BUY_B_SELL_A"
                        cheap_market = corr.market_b_id
                        expensive_market = corr.market_a_id

                    # Get orderbooks for liquidity assessment
                    ob_a = self.monitor.get_current_orderbook(corr.market_a_id)
                    ob_b = self.monitor.get_current_orderbook(corr.market_b_id)

                    max_size = 0.0
                    profit_potential = spread

                    if ob_a and ob_b:
                        # Assess two-sided liquidity
                        side_a = "buy" if pa < pb else "sell"
                        side_b = "sell" if pa < pb else "buy"
                        two_sided = self.liquidity.assess_two_sided_liquidity(
                            ob_a, side_a, ob_b, side_b
                        )
                        max_size = two_sided.executable_size
                        if max_size > 0:
                            profit_potential = two_sided.profit_per_dollar

                    # Check minimum liquidity
                    if max_size < self.config.min_liquidity_threshold:
                        continue

                    results.append(Divergence(
                        id=Divergence.generate_id(),
                        divergence_type=DivergenceType.PRICE_SPREAD,
                        detected_at=datetime.utcnow(),
                        market_ids=[corr.market_a_id, corr.market_b_id],
                        correlation_id=None,
                        current_prices={corr.market_a_id: pa, corr.market_b_id: pb},
                        current_orderbooks={
                            corr.market_a_id: ob_a,
                            corr.market_b_id: ob_b
                        } if ob_a and ob_b else {},
                        expected_relationship="A ≈ B",
                        expected_value=pa,
                        actual_value=pb,
                        divergence_amount=spread,
                        divergence_pct=spread / max(pa, pb, 0.001),
                        direction=direction,
                        is_arbitrage=corr.correlation_type == CorrelationType.EQUIVALENT,
                        profit_potential=profit_potential,
                        max_executable_size=max_size,
                        confidence=corr.confidence,
                        supporting_evidence=[
                            f"Price spread: {spread:.4f}",
                            f"Correlation type: {corr.correlation_type.value}",
                        ],
                    ))

            except Exception as e:
                logger.error(f"Error checking spread for {corr.market_a_id}/{corr.market_b_id}: {e}")

        return results

    async def detect_threshold_violations(
        self,
        rules: List[LogicalRule],
    ) -> List[Divergence]:
        """
        Detect threshold ordering violations.

        For THRESHOLD_ORDERING rules (e.g., BTC > $90K implies BTC > $100K),
        checks if prices respect the ordering.

        Example:
        - "BTC > $90K" at 45¢
        - "BTC > $100K" at 48¢
        - VIOLATION: higher threshold should be cheaper
        - Guaranteed profit: buy $90K, sell $100K
        """
        results = []

        threshold_rules = [r for r in rules if r.rule_type == LogicalRuleType.THRESHOLD_ORDERING]

        for rule in threshold_rules:
            try:
                # Get market IDs from metadata
                low_id = rule.metadata.get("lower_strike_market")
                high_id = rule.metadata.get("higher_strike_market")

                if not low_id or not high_id:
                    continue

                p_low = self.monitor.get_current_price(low_id)
                p_high = self.monitor.get_current_price(high_id)

                if p_low is None or p_high is None:
                    continue

                # Check violation: higher threshold should have LOWER price
                # P(BTC > $100K) <= P(BTC > $90K)
                violation_amount = p_high - p_low + rule.tolerance

                if violation_amount > 0:
                    # This is a guaranteed arbitrage
                    ob_low = self.monitor.get_current_orderbook(low_id)
                    ob_high = self.monitor.get_current_orderbook(high_id)

                    max_size = 0.0
                    if ob_low and ob_high:
                        two_sided = self.liquidity.assess_two_sided_liquidity(
                            ob_low, "buy", ob_high, "sell"
                        )
                        max_size = two_sided.executable_size

                    results.append(Divergence(
                        id=Divergence.generate_id(),
                        divergence_type=DivergenceType.THRESHOLD_VIOLATION,
                        detected_at=datetime.utcnow(),
                        market_ids=[low_id, high_id],
                        rule_id=None,
                        current_prices={low_id: p_low, high_id: p_high},
                        current_orderbooks={
                            low_id: ob_low,
                            high_id: ob_high
                        } if ob_low and ob_high else {},
                        expected_relationship="P(Low) >= P(High)",
                        expected_value=p_low,
                        actual_value=p_high,
                        divergence_amount=violation_amount,
                        divergence_pct=violation_amount / max(p_low, 0.001),
                        direction="BUY_LOW_SELL_HIGH",
                        is_arbitrage=True,  # Guaranteed profit
                        profit_potential=violation_amount,
                        max_executable_size=max_size,
                        confidence=1.0,
                        supporting_evidence=[
                            f"Rule: {rule.constraint_desc}",
                            f"Violation: {p_high:.4f} > {p_low:.4f}",
                            "Guaranteed arbitrage opportunity",
                        ],
                    ))

            except Exception as e:
                logger.error(f"Error checking threshold rule: {e}")

        return results

    async def detect_inverse_sum_divergence(
        self,
        rules: List[LogicalRule],
    ) -> List[Divergence]:
        """
        Detect when mutually exclusive outcomes don't sum to ~100%.

        For MUTUALLY_EXCLUSIVE or EXHAUSTIVE rules, checks if YES prices
        of all outcomes sum to approximately 100%.

        Example:
        - Trump YES: 52¢
        - Biden YES: 44¢
        - Other YES: 8¢
        - Sum = 104¢ (should be ~100¢)
        - Opportunity: sell overpriced outcome(s)
        """
        results = []
        threshold = self.config.get_threshold(DivergenceType.INVERSE_SUM)

        # Filter to mutually exclusive / exhaustive rules
        sum_rules = [
            r for r in rules
            if r.rule_type in (LogicalRuleType.MUTUALLY_EXCLUSIVE, LogicalRuleType.EXHAUSTIVE)
        ]

        for rule in sum_rules:
            try:
                # Get prices for all markets in the rule
                prices = {}
                missing = False

                for mid in rule.market_ids:
                    p = self.monitor.get_current_price(mid)
                    if p is None:
                        missing = True
                        break
                    prices[mid] = p

                if missing:
                    continue

                # Calculate sum of YES prices
                total = sum(prices.values())
                expected = 1.0  # 100 cents

                deviation = abs(total - expected)

                if deviation > threshold:
                    # Determine direction
                    if total > expected:
                        direction = "SELL_OVERPRICED"
                        evidence = f"Sum {total:.4f} > 1.0, sell overpriced outcomes"
                    else:
                        direction = "BUY_UNDERPRICED"
                        evidence = f"Sum {total:.4f} < 1.0, buy underpriced outcomes"

                    # Identify the most mispriced market
                    if total > expected:
                        # Find highest priced (most overpriced)
                        mispriced_id = max(prices.keys(), key=lambda k: prices[k])
                    else:
                        # Find lowest priced (most underpriced)
                        mispriced_id = min(prices.keys(), key=lambda k: prices[k])

                    # Get orderbooks
                    orderbooks = {}
                    for mid in rule.market_ids:
                        ob = self.monitor.get_current_orderbook(mid)
                        if ob:
                            orderbooks[mid] = ob

                    results.append(Divergence(
                        id=Divergence.generate_id(),
                        divergence_type=DivergenceType.INVERSE_SUM,
                        detected_at=datetime.utcnow(),
                        market_ids=rule.market_ids,
                        rule_id=None,
                        current_prices=prices,
                        current_orderbooks=orderbooks,
                        expected_relationship="Sum ≈ 1.0",
                        expected_value=expected,
                        actual_value=total,
                        divergence_amount=deviation,
                        divergence_pct=deviation / expected,
                        direction=direction,
                        is_arbitrage=True,  # Can construct riskless trade
                        profit_potential=deviation,
                        max_executable_size=0.0,  # Need complex calculation
                        confidence=rule.confidence,
                        supporting_evidence=[
                            f"Rule: {rule.constraint_desc}",
                            evidence,
                            f"Most mispriced: {mispriced_id} at {prices[mispriced_id]:.4f}",
                        ],
                        metadata={
                            "price_sum": total,
                            "most_mispriced_market": mispriced_id,
                        },
                    ))

            except Exception as e:
                logger.error(f"Error checking inverse sum rule: {e}")

        return results

    async def detect_lagging_market(
        self,
        correlations: List[MarketCorrelation],
        lookback_seconds: Optional[int] = None,
    ) -> List[Divergence]:
        """
        Detect when one market has moved but a correlated market hasn't.

        For CAUSAL/EQUIVALENT correlations, checks if one market moved
        recently while the other hasn't caught up.

        Example:
        - "Trump wins PA" jumped 52¢ → 60¢ in last 5 min
        - "Trump wins national" still at 54¢
        - Expected: national should have moved ~4¢ up
        - Opportunity: buy national
        """
        results = []
        lookback = lookback_seconds or self.config.lagging_lookback_seconds
        threshold = self.config.get_threshold(DivergenceType.LAGGING_MARKET)

        # Filter to causal or equivalent correlations
        candidates = [
            c for c in correlations
            if c.confidence >= self.config.min_correlation_confidence
            and c.correlation_type in (
                CorrelationType.CAUSAL,
                CorrelationType.EQUIVALENT,
                CorrelationType.MATHEMATICAL,
            )
        ]

        for corr in candidates:
            try:
                # Get price changes for both markets
                old_a, new_a = self.monitor.get_price_change(corr.market_a_id, lookback)
                old_b, new_b = self.monitor.get_price_change(corr.market_b_id, lookback)

                if None in (old_a, new_a, old_b, new_b):
                    continue

                change_a = new_a - old_a
                change_b = new_b - old_b

                # Check if one moved significantly and the other didn't
                min_movement = 0.03  # 3 cents minimum movement
                max_lag = 0.01  # 1 cent maximum lag

                lagging_market = None
                leader_market = None
                leader_change = 0.0

                if abs(change_a) > min_movement and abs(change_b) < max_lag:
                    leader_market = corr.market_a_id
                    lagging_market = corr.market_b_id
                    leader_change = change_a
                elif abs(change_b) > min_movement and abs(change_a) < max_lag:
                    leader_market = corr.market_b_id
                    lagging_market = corr.market_a_id
                    leader_change = change_b

                if lagging_market is None:
                    continue

                # Calculate expected movement
                hist_corr = corr.historical_correlation or corr.confidence
                expected_change = leader_change * hist_corr
                current_lag = abs(expected_change)

                if current_lag < threshold:
                    continue

                # Determine trade direction
                if leader_change > 0:
                    direction = f"BUY_{lagging_market}"
                else:
                    direction = f"SELL_{lagging_market}"

                # Get orderbooks
                ob_lagging = self.monitor.get_current_orderbook(lagging_market)
                max_size = 0.0
                if ob_lagging:
                    side = "buy" if leader_change > 0 else "sell"
                    max_size = self.liquidity.get_executable_size(ob_lagging, side)

                results.append(Divergence(
                    id=Divergence.generate_id(),
                    divergence_type=DivergenceType.LAGGING_MARKET,
                    detected_at=datetime.utcnow(),
                    market_ids=[leader_market, lagging_market],
                    correlation_id=None,
                    current_prices={
                        corr.market_a_id: new_a,
                        corr.market_b_id: new_b,
                    },
                    current_orderbooks={lagging_market: ob_lagging} if ob_lagging else {},
                    expected_relationship=f"{leader_market} leads {lagging_market}",
                    expected_value=expected_change,
                    actual_value=change_b if lagging_market == corr.market_b_id else change_a,
                    divergence_amount=current_lag,
                    divergence_pct=current_lag / max(abs(leader_change), 0.001),
                    direction=direction,
                    is_arbitrage=False,  # Not guaranteed
                    profit_potential=current_lag,
                    max_executable_size=max_size,
                    confidence=corr.confidence * 0.8,  # Discount for uncertainty
                    supporting_evidence=[
                        f"Leader ({leader_market}) moved {leader_change:.4f}",
                        f"Follower ({lagging_market}) hasn't caught up",
                        f"Expected change: {expected_change:.4f}",
                    ],
                    metadata={
                        "leader_market": leader_market,
                        "lagging_market": lagging_market,
                        "leader_change": leader_change,
                    },
                ))

            except Exception as e:
                logger.error(f"Error checking lagging market: {e}")

        return results

    async def detect_correlation_break(
        self,
        correlations: List[MarketCorrelation],
        window_minutes: Optional[int] = None,
    ) -> List[Divergence]:
        """
        Detect when historical correlation has broken down.

        For statistically correlated markets, compares recent correlation
        to historical correlation and flags significant deviations.

        Example:
        - Historical correlation: 0.85
        - Last hour correlation: 0.20
        - Something changed, investigate
        """
        results = []
        window = window_minutes or self.config.correlation_break_window_minutes
        lookback_seconds = window * 60

        candidates = [
            c for c in correlations
            if c.historical_correlation is not None
            and c.confidence >= 0.5  # Lower threshold for monitoring
        ]

        for corr in candidates:
            try:
                # Get price histories
                hist_a = self.monitor.get_price_history(corr.market_a_id, lookback_seconds)
                hist_b = self.monitor.get_price_history(corr.market_b_id, lookback_seconds)

                if len(hist_a) < 10 or len(hist_b) < 10:
                    continue

                # Align timestamps (simple approach: just use the prices)
                prices_a = [p for _, p in hist_a]
                prices_b = [p for _, p in hist_b]

                # Use minimum length
                min_len = min(len(prices_a), len(prices_b))
                prices_a = prices_a[-min_len:]
                prices_b = prices_b[-min_len:]

                # Calculate recent correlation
                if len(prices_a) < 5:
                    continue

                try:
                    recent_corr = np.corrcoef(prices_a, prices_b)[0, 1]
                except Exception:
                    continue

                if np.isnan(recent_corr):
                    continue

                # Compare to historical
                hist_corr = corr.historical_correlation
                corr_diff = abs(recent_corr - hist_corr)

                # Threshold for correlation break
                break_threshold = 0.3  # 0.3 correlation difference

                if corr_diff > break_threshold:
                    current_a = self.monitor.get_current_price(corr.market_a_id)
                    current_b = self.monitor.get_current_price(corr.market_b_id)

                    results.append(Divergence(
                        id=Divergence.generate_id(),
                        divergence_type=DivergenceType.CORRELATION_BREAK,
                        detected_at=datetime.utcnow(),
                        market_ids=[corr.market_a_id, corr.market_b_id],
                        correlation_id=None,
                        current_prices={
                            corr.market_a_id: current_a or 0.0,
                            corr.market_b_id: current_b or 0.0,
                        },
                        expected_relationship=f"Correlation ≈ {hist_corr:.2f}",
                        expected_value=hist_corr,
                        actual_value=recent_corr,
                        divergence_amount=corr_diff,
                        divergence_pct=corr_diff / max(abs(hist_corr), 0.001),
                        direction="INVESTIGATE",
                        is_arbitrage=False,
                        profit_potential=0.0,  # Unknown
                        max_executable_size=0.0,
                        confidence=0.5,  # Uncertain
                        supporting_evidence=[
                            f"Historical correlation: {hist_corr:.3f}",
                            f"Recent correlation: {recent_corr:.3f}",
                            f"Window: {window} minutes",
                            "Relationship may have changed",
                        ],
                        metadata={
                            "historical_correlation": hist_corr,
                            "recent_correlation": recent_corr,
                            "window_minutes": window,
                        },
                    ))

            except Exception as e:
                logger.error(f"Error checking correlation break: {e}")

        return results

    async def detect_lead_lag_opportunity(
        self,
        correlations: List[MarketCorrelation],
    ) -> List[Divergence]:
        """
        Detect lead-lag opportunities where leader moved but follower hasn't.

        For correlations with known lead-lag relationships, checks if the
        leader has moved but the follower hasn't yet responded.

        Example:
        - Market A leads Market B by ~30 seconds
        - A moved from 50¢ → 55¢ 15 seconds ago
        - B still at 50¢
        - Opportunity: buy B before it catches up
        """
        results = []

        # Filter to correlations with lead-lag metadata
        candidates = [
            c for c in correlations
            if c.metadata.get("lead_lag_seconds") is not None
            and c.metadata.get("leader_market_id") is not None
        ]

        for corr in candidates:
            try:
                lag_seconds = corr.metadata["lead_lag_seconds"]
                leader_id = corr.metadata["leader_market_id"]
                follower_id = (
                    corr.market_b_id if leader_id == corr.market_a_id
                    else corr.market_a_id
                )

                # Look at leader movement over lag period
                old_leader, new_leader = self.monitor.get_price_change(leader_id, lag_seconds)
                old_follower, new_follower = self.monitor.get_price_change(follower_id, lag_seconds)

                if None in (old_leader, new_leader, old_follower, new_follower):
                    continue

                leader_change = abs(new_leader - old_leader)
                follower_change = abs(new_follower - old_follower)

                # Threshold for significant movement
                min_leader_move = 0.03  # 3 cents
                max_follower_move = 0.01  # 1 cent

                if leader_change >= min_leader_move and follower_change < max_follower_move:
                    # Leader moved, follower hasn't
                    direction_sign = 1 if new_leader > old_leader else -1
                    direction = f"{'BUY' if direction_sign > 0 else 'SELL'}_{follower_id}"

                    ob_follower = self.monitor.get_current_orderbook(follower_id)
                    max_size = 0.0
                    if ob_follower:
                        side = "buy" if direction_sign > 0 else "sell"
                        max_size = self.liquidity.get_executable_size(ob_follower, side)

                    results.append(Divergence(
                        id=Divergence.generate_id(),
                        divergence_type=DivergenceType.LEAD_LAG_OPPORTUNITY,
                        detected_at=datetime.utcnow(),
                        market_ids=[leader_id, follower_id],
                        correlation_id=None,
                        current_prices={
                            leader_id: new_leader,
                            follower_id: new_follower,
                        },
                        current_orderbooks={follower_id: ob_follower} if ob_follower else {},
                        expected_relationship=f"{leader_id} leads {follower_id} by ~{lag_seconds}s",
                        expected_value=new_leader,
                        actual_value=new_follower,
                        divergence_amount=leader_change,
                        divergence_pct=leader_change / max(old_leader, 0.001),
                        direction=direction,
                        is_arbitrage=False,  # Statistical, not guaranteed
                        profit_potential=leader_change * corr.confidence,
                        max_executable_size=max_size,
                        confidence=corr.confidence * 0.9,
                        supporting_evidence=[
                            f"Leader ({leader_id}) moved: {old_leader:.4f} → {new_leader:.4f}",
                            f"Follower ({follower_id}) static at {new_follower:.4f}",
                            f"Lag period: {lag_seconds}s",
                        ],
                        metadata={
                            "leader_market_id": leader_id,
                            "follower_market_id": follower_id,
                            "lag_seconds": lag_seconds,
                            "leader_change": new_leader - old_leader,
                        },
                    ))

            except Exception as e:
                logger.error(f"Error checking lead-lag: {e}")

        return results

    # --- Synchronous Helpers (backward compatibility) ---

    def _check_spreads_sync(self, correlations: List[MarketCorrelation]) -> List[Divergence]:
        """Synchronous spread check."""
        results = []
        threshold = self.config.min_divergence_threshold

        for corr in correlations:
            pa = self.monitor.get_current_price(corr.market_a_id)
            pb = self.monitor.get_current_price(corr.market_b_id)

            if pa is None or pb is None:
                continue

            spread = abs(pa - pb)
            if spread > threshold:
                results.append(Divergence(
                    id=Divergence.generate_id(),
                    divergence_type=DivergenceType.PRICE_SPREAD,
                    detected_at=datetime.utcnow(),
                    market_ids=[corr.market_a_id, corr.market_b_id],
                    current_prices={corr.market_a_id: pa, corr.market_b_id: pb},
                    expected_relationship="A ≈ B",
                    expected_value=pa,
                    actual_value=pb,
                    divergence_amount=spread,
                    divergence_pct=spread / max(pa, 0.001),
                    profit_potential=spread,
                    confidence=corr.confidence,
                ))
        return results

    def _check_rule_violations_sync(self, rules: List[LogicalRule]) -> List[Divergence]:
        """Synchronous rule violation check."""
        results = []

        for rule in rules:
            if rule.rule_type != LogicalRuleType.THRESHOLD_ORDERING:
                continue

            low_id = rule.metadata.get("lower_strike_market")
            high_id = rule.metadata.get("higher_strike_market")

            if not low_id or not high_id:
                continue

            p_low = self.monitor.get_current_price(low_id)
            p_high = self.monitor.get_current_price(high_id)

            if p_low is None or p_high is None:
                continue

            if p_high > p_low + rule.tolerance:
                results.append(Divergence(
                    id=Divergence.generate_id(),
                    divergence_type=DivergenceType.THRESHOLD_VIOLATION,
                    detected_at=datetime.utcnow(),
                    market_ids=[low_id, high_id],
                    current_prices={low_id: p_low, high_id: p_high},
                    expected_relationship="P(Low) >= P(High)",
                    divergence_amount=p_high - p_low,
                    is_arbitrage=True,
                    profit_potential=p_high - p_low,
                    confidence=1.0,
                ))
        return results

    def _check_lead_lag_sync(self, correlations: List[MarketCorrelation]) -> List[Divergence]:
        """Synchronous lead-lag check."""
        results = []

        for corr in correlations:
            lag = corr.metadata.get("lead_lag_seconds")
            leader = corr.metadata.get("leader_market_id")

            if not lag or not leader:
                continue

            follower = (
                corr.market_b_id if leader == corr.market_a_id
                else corr.market_a_id
            )

            old_leader, new_leader = self.monitor.get_price_change(leader, lag)
            old_follower, new_follower = self.monitor.get_price_change(follower, lag)

            if None in (old_leader, new_leader, old_follower, new_follower):
                continue

            leader_change = abs(new_leader - old_leader)
            follower_change = abs(new_follower - old_follower)

            if leader_change > 0.05 and follower_change < 0.01:
                results.append(Divergence(
                    id=Divergence.generate_id(),
                    divergence_type=DivergenceType.LEAD_LAG_OPPORTUNITY,
                    detected_at=datetime.utcnow(),
                    market_ids=[leader, follower],
                    current_prices={leader: new_leader, follower: new_follower},
                    expected_relationship=f"{leader} leads {follower}",
                    divergence_amount=leader_change,
                    confidence=corr.confidence,
                ))

        return results

    # --- Utility Methods ---

    def _deduplicate_divergences(self, divergences: List[Divergence]) -> List[Divergence]:
        """Remove duplicate divergences (same markets, same type)."""
        seen: Set[str] = set()
        unique = []

        for div in divergences:
            key = f"{div.divergence_type.value}:{':'.join(sorted(div.market_ids))}"
            if key not in seen:
                seen.add(key)
                unique.append(div)

        return unique

    def _apply_debouncing(self, divergences: List[Divergence]) -> List[Divergence]:
        """Filter out divergences that were recently detected."""
        now = time.time()
        debounce_time = self.config.debounce_seconds
        result = []

        for div in divergences:
            key = f"{div.divergence_type.value}:{':'.join(sorted(div.market_ids))}"
            last_time = self._last_detection.get(key, 0)

            if now - last_time >= debounce_time:
                self._last_detection[key] = now
                result.append(div)

        return result

    def _cleanup_expired(self) -> None:
        """Remove expired divergences from active tracking."""
        now = datetime.utcnow()
        expired_ids = [
            div_id for div_id, div in self._active_divergences.items()
            if div.expires_at and div.expires_at < now
        ]

        for div_id in expired_ids:
            div = self._active_divergences.pop(div_id)
            div.mark_expired()

    # --- State Access ---

    def get_active_divergences(self) -> List[Divergence]:
        """Get all currently active divergences."""
        return list(self._active_divergences.values())

    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            "total_detections": self._detection_count,
            "active_divergences": len(self._active_divergences),
            "false_positives": self._false_positive_count,
        }

    def mark_false_positive(self, divergence_id: str, reason: str = "") -> bool:
        """Mark a divergence as a false positive."""
        if divergence_id in self._active_divergences:
            self._active_divergences[divergence_id].mark_false_positive(reason)
            self._false_positive_count += 1
            return True
        return False

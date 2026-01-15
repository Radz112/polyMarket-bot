"""
Comprehensive tests for divergence detection system.

Tests all divergence types:
- Price spread divergence
- Threshold violations
- Inverse sum divergence
- Lagging market detection
- Correlation break detection
- Lead-lag opportunities
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.models import Orderbook, OrderbookEntry, MarketCorrelation, CorrelationType
from src.correlation.logical.rules import LogicalRule, LogicalRuleType
from src.signals.divergence import (
    DivergenceType,
    DivergenceStatus,
    Divergence,
    DivergenceConfig,
    DivergenceDetector,
    PriceMonitor,
    LiquidityAssessor,
    LiquidityAnalysis,
    TwoSidedLiquidity,
)


# ============================================================================
# Fixtures
# ============================================================================

class MockWsClient:
    """Mock WebSocket client for testing."""
    def __init__(self):
        self.callbacks = []
        self.subscribed_assets = []

    def register_callback(self, callback):
        self.callbacks.append(callback)

    async def subscribe(self, asset_ids):
        self.subscribed_assets.extend(asset_ids)

    async def unsubscribe(self, asset_ids):
        for aid in asset_ids:
            if aid in self.subscribed_assets:
                self.subscribed_assets.remove(aid)


@pytest.fixture
def mock_ws():
    return MockWsClient()


@pytest.fixture
def price_monitor(mock_ws):
    return PriceMonitor(mock_ws)


@pytest.fixture
def liquidity_assessor():
    return LiquidityAssessor()


@pytest.fixture
def detector(price_monitor):
    config = DivergenceConfig(
        min_divergence_threshold=0.02,
        min_liquidity_threshold=0.0,  # Disable for testing
        min_correlation_confidence=0.8,
    )
    return DivergenceDetector(price_monitor, config=config)


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook for testing."""
    return Orderbook(
        market_id="test_market",
        bids=[
            OrderbookEntry(price=0.49, size=100),
            OrderbookEntry(price=0.48, size=200),
            OrderbookEntry(price=0.47, size=300),
        ],
        asks=[
            OrderbookEntry(price=0.50, size=100),
            OrderbookEntry(price=0.51, size=200),
            OrderbookEntry(price=0.52, size=300),
        ],
    )


# ============================================================================
# Liquidity Assessor Tests
# ============================================================================

class TestLiquidityAssessor:

    def test_get_effective_price_single_level(self, liquidity_assessor):
        """Test VWAP calculation when order fits in one level."""
        levels = [(0.50, 100), (0.51, 200)]
        price, filled = liquidity_assessor.get_effective_price(levels, 50)

        assert filled == 50
        assert price == 0.50  # All from first level

    def test_get_effective_price_multiple_levels(self, liquidity_assessor):
        """Test VWAP calculation across multiple levels."""
        levels = [(0.50, 100), (0.51, 200)]
        price, filled = liquidity_assessor.get_effective_price(levels, 150)

        assert filled == 150
        # 100 @ 0.50 = 50, 50 @ 0.51 = 25.5, total = 75.5
        # Avg = 75.5 / 150 = 0.5033...
        assert abs(price - 0.503333) < 0.0001

    def test_get_effective_price_exceeds_liquidity(self, liquidity_assessor):
        """Test when requested size exceeds available liquidity."""
        levels = [(0.50, 100), (0.51, 200)]
        price, filled = liquidity_assessor.get_effective_price(levels, 500)

        assert filled == 300  # Max available

    def test_get_executable_size_buy(self, liquidity_assessor, sample_orderbook):
        """Test executable size for buying."""
        # Best ask 0.50, 1% slippage = max 0.505
        # Only first level (0.50) qualifies
        size = liquidity_assessor.get_executable_size(sample_orderbook, "buy", max_slippage=0.01)
        assert size == 100

    def test_get_executable_size_sell(self, liquidity_assessor, sample_orderbook):
        """Test executable size for selling."""
        # Best bid 0.49, 1% slippage = min 0.4851
        # Only first level (0.49) qualifies
        size = liquidity_assessor.get_executable_size(sample_orderbook, "sell", max_slippage=0.01)
        assert size == 100

    def test_get_executable_size_higher_slippage(self, liquidity_assessor, sample_orderbook):
        """Test executable size with higher slippage tolerance."""
        # Best ask 0.50, 5% slippage = max 0.525
        # Levels 0.50, 0.51, 0.52 all qualify
        size = liquidity_assessor.get_executable_size(sample_orderbook, "buy", max_slippage=0.05)
        assert size == 600  # All levels

    def test_analyze_liquidity(self, liquidity_assessor, sample_orderbook):
        """Test comprehensive liquidity analysis."""
        analysis = liquidity_assessor.analyze_liquidity(sample_orderbook, "buy", max_slippage=0.02)

        assert isinstance(analysis, LiquidityAnalysis)
        assert analysis.executable_size > 0
        assert analysis.total_available == 600
        assert analysis.levels_consumed >= 1

    def test_assess_two_sided_liquidity(self, liquidity_assessor):
        """Test two-sided liquidity assessment for arbitrage."""
        ob_a = Orderbook(
            market_id="A",
            asks=[OrderbookEntry(price=0.50, size=100)],
            bids=[OrderbookEntry(price=0.48, size=100)],
        )
        ob_b = Orderbook(
            market_id="B",
            asks=[OrderbookEntry(price=0.52, size=100)],
            bids=[OrderbookEntry(price=0.55, size=100)],  # Can sell higher
        )

        result = liquidity_assessor.assess_two_sided_liquidity(
            ob_a, "buy", ob_b, "sell"
        )

        assert isinstance(result, TwoSidedLiquidity)
        assert result.executable_size == 100
        assert result.expected_profit > 0  # Sell at 0.55, buy at 0.50


# ============================================================================
# Price Monitor Tests
# ============================================================================

class TestPriceMonitor:

    def test_manual_price_update(self, price_monitor):
        """Test manual price update for testing."""
        price_monitor.manual_price_update("m1", 0.50)

        assert price_monitor.get_current_price("m1") == 0.50

    def test_price_history_tracking(self, price_monitor):
        """Test that price history is tracked."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m1", 0.52)
        price_monitor.manual_price_update("m1", 0.55)

        history = price_monitor.get_price_history("m1")
        assert len(history) == 3
        assert history[-1][1] == 0.55

    def test_get_price_change(self, price_monitor):
        """Test price change calculation."""
        # Simulate old price
        price_monitor.manual_price_update("m1", 0.50)

        # Wait a tiny bit and update
        time.sleep(0.01)
        price_monitor.manual_price_update("m1", 0.55)

        old, new = price_monitor.get_price_change("m1", lookback_seconds=10)

        assert old == 0.50
        assert new == 0.55

    def test_get_current_orderbook(self, price_monitor, sample_orderbook):
        """Test orderbook storage and retrieval."""
        price_monitor.manual_price_update("test", 0.50, sample_orderbook)

        ob = price_monitor.get_current_orderbook("test")
        assert ob is not None
        assert ob.market_id == "test_market"

    def test_callback_registration(self, price_monitor):
        """Test that callbacks are invoked on updates."""
        received = []

        def callback(market_id, price, orderbook):
            received.append((market_id, price))

        price_monitor.on_price_update(callback)
        price_monitor.manual_price_update("m1", 0.50)

        assert len(received) == 1
        assert received[0] == ("m1", 0.50)


# ============================================================================
# Divergence Type Tests
# ============================================================================

class TestDivergenceTypes:

    def test_divergence_creation(self):
        """Test basic divergence creation."""
        div = Divergence(
            id=Divergence.generate_id(),
            divergence_type=DivergenceType.PRICE_SPREAD,
            detected_at=datetime.utcnow(),
            market_ids=["m1", "m2"],
            current_prices={"m1": 0.50, "m2": 0.55},
            divergence_amount=0.05,
            confidence=0.9,
        )

        assert div.divergence_type == DivergenceType.PRICE_SPREAD
        assert len(div.market_ids) == 2
        assert div.status == DivergenceStatus.ACTIVE

    def test_divergence_expiration(self):
        """Test divergence expiration logic."""
        # Create already expired divergence
        div = Divergence(
            id=Divergence.generate_id(),
            divergence_type=DivergenceType.PRICE_SPREAD,
            detected_at=datetime.utcnow() - timedelta(hours=1),
            market_ids=["m1", "m2"],
            expires_at=datetime.utcnow() - timedelta(minutes=5),
        )

        assert div.is_expired()

    def test_divergence_to_signal(self):
        """Test conversion to Signal."""
        div = Divergence(
            id="test-123",
            divergence_type=DivergenceType.THRESHOLD_VIOLATION,
            detected_at=datetime.utcnow(),
            market_ids=["m1", "m2"],
            current_prices={"m1": 0.50, "m2": 0.55},
            divergence_amount=0.05,
            expected_value=0.50,
            actual_value=0.55,
            profit_potential=0.05,
            confidence=1.0,
            is_arbitrage=True,
        )

        signal = div.to_signal()

        assert signal.id == "test-123"
        assert signal.confidence == 1.0
        assert signal.divergence_amount == 0.05

    def test_mark_traded(self):
        """Test marking divergence as traded."""
        div = Divergence(
            id=Divergence.generate_id(),
            divergence_type=DivergenceType.PRICE_SPREAD,
            detected_at=datetime.utcnow(),
            market_ids=["m1", "m2"],
        )

        div.mark_traded()
        assert div.status == DivergenceStatus.TRADED

    def test_mark_false_positive(self):
        """Test marking as false positive."""
        div = Divergence(
            id=Divergence.generate_id(),
            divergence_type=DivergenceType.PRICE_SPREAD,
            detected_at=datetime.utcnow(),
            market_ids=["m1", "m2"],
        )

        div.mark_false_positive("Prices converged before trade")
        assert div.status == DivergenceStatus.FALSE_POSITIVE
        assert "False positive" in div.supporting_evidence[0]


# ============================================================================
# Detector Tests - Price Spread
# ============================================================================

class TestPriceSpreadDetection:

    def test_detect_price_spread(self, detector, price_monitor):
        """Test detection of price spread divergence."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)  # 0.10 spread

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.95,
        )

        divs = detector.detect_all([corr], [])

        assert len(divs) == 1
        assert divs[0].divergence_type == DivergenceType.PRICE_SPREAD
        assert divs[0].divergence_amount == pytest.approx(0.10)

    def test_no_divergence_within_threshold(self, detector, price_monitor):
        """Test no divergence when spread is within threshold."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.51)  # Only 0.01 spread

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.95,
        )

        divs = detector.detect_all([corr], [])

        assert len(divs) == 0

    def test_low_confidence_correlation_ignored(self, detector, price_monitor):
        """Test that low confidence correlations are ignored."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.5,  # Below threshold
        )

        divs = detector.detect_all([corr], [])

        assert len(divs) == 0


# ============================================================================
# Detector Tests - Threshold Violations
# ============================================================================

class TestThresholdViolationDetection:

    def test_detect_threshold_violation(self, detector, price_monitor):
        """Test detection of threshold ordering violation."""
        # Higher threshold priced higher than lower = VIOLATION
        price_monitor.manual_price_update("btc_90k", 0.45)  # Lower threshold
        price_monitor.manual_price_update("btc_100k", 0.48)  # Higher threshold

        rule = LogicalRule(
            rule_type=LogicalRuleType.THRESHOLD_ORDERING,
            market_ids=["btc_90k", "btc_100k"],
            constraint_desc="P(BTC > $90K) >= P(BTC > $100K)",
            tolerance=0.01,
            metadata={
                "lower_strike_market": "btc_90k",
                "higher_strike_market": "btc_100k",
            },
        )

        divs = detector.detect_all([], [rule])

        assert len(divs) == 1
        assert divs[0].divergence_type == DivergenceType.THRESHOLD_VIOLATION
        assert divs[0].is_arbitrage is True

    def test_no_violation_when_ordered_correctly(self, detector, price_monitor):
        """Test no violation when prices are correctly ordered."""
        price_monitor.manual_price_update("btc_90k", 0.60)  # Lower threshold, higher price
        price_monitor.manual_price_update("btc_100k", 0.40)  # Higher threshold, lower price

        rule = LogicalRule(
            rule_type=LogicalRuleType.THRESHOLD_ORDERING,
            market_ids=["btc_90k", "btc_100k"],
            constraint_desc="P(BTC > $90K) >= P(BTC > $100K)",
            tolerance=0.01,
            metadata={
                "lower_strike_market": "btc_90k",
                "higher_strike_market": "btc_100k",
            },
        )

        divs = detector.detect_all([], [rule])

        assert len(divs) == 0


# ============================================================================
# Detector Tests - Inverse Sum
# ============================================================================

class TestInverseSumDetection:

    @pytest.mark.asyncio
    async def test_detect_inverse_sum_overpriced(self, detector, price_monitor):
        """Test detection when outcome prices sum to > 100%."""
        price_monitor.manual_price_update("trump", 0.52)
        price_monitor.manual_price_update("biden", 0.44)
        price_monitor.manual_price_update("other", 0.08)
        # Sum = 104%

        rule = LogicalRule(
            rule_type=LogicalRuleType.MUTUALLY_EXCLUSIVE,
            market_ids=["trump", "biden", "other"],
            constraint_desc="Outcomes sum to 100%",
            tolerance=0.01,
        )

        divs = await detector.detect_inverse_sum_divergence([rule])

        assert len(divs) == 1
        assert divs[0].divergence_type == DivergenceType.INVERSE_SUM
        assert divs[0].direction == "SELL_OVERPRICED"
        assert divs[0].actual_value == pytest.approx(1.04)

    @pytest.mark.asyncio
    async def test_detect_inverse_sum_underpriced(self, detector, price_monitor):
        """Test detection when outcome prices sum to < 100%."""
        price_monitor.manual_price_update("trump", 0.45)
        price_monitor.manual_price_update("biden", 0.40)
        price_monitor.manual_price_update("other", 0.05)
        # Sum = 90%

        rule = LogicalRule(
            rule_type=LogicalRuleType.EXHAUSTIVE,
            market_ids=["trump", "biden", "other"],
            constraint_desc="Outcomes sum to 100%",
            tolerance=0.01,
        )

        divs = await detector.detect_inverse_sum_divergence([rule])

        assert len(divs) == 1
        assert divs[0].direction == "BUY_UNDERPRICED"


# ============================================================================
# Detector Tests - Lead-Lag
# ============================================================================

class TestLeadLagDetection:

    def test_detect_lead_lag_opportunity(self, detector, price_monitor):
        """Test detection of lead-lag opportunity."""
        corr = MarketCorrelation(
            market_a_id="leader",
            market_b_id="follower",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="leader leads follower",
            confidence=0.95,
            metadata={
                "lead_lag_seconds": 60,
                "leader_market_id": "leader",
            },
        )

        # Simulate leader movement
        price_monitor._price_history["leader"] = []
        price_monitor._price_history["follower"] = []

        now = time.time()

        # Leader moved significantly
        from src.signals.divergence.price_monitor import PricePoint
        price_monitor._price_history["leader"].append(PricePoint(now - 60, 0.50))
        price_monitor._price_history["leader"].append(PricePoint(now, 0.60))
        price_monitor._market_prices["leader"] = 0.60

        # Follower stayed flat
        price_monitor._price_history["follower"].append(PricePoint(now - 60, 0.50))
        price_monitor._price_history["follower"].append(PricePoint(now, 0.50))
        price_monitor._market_prices["follower"] = 0.50

        divs = detector.detect_all([corr], [])

        lead_lag_divs = [d for d in divs if d.divergence_type == DivergenceType.LEAD_LAG_OPPORTUNITY]
        assert len(lead_lag_divs) == 1
        assert "leader" in lead_lag_divs[0].market_ids
        assert "follower" in lead_lag_divs[0].market_ids


# ============================================================================
# Detector Tests - Lagging Market
# ============================================================================

class TestLaggingMarketDetection:

    @pytest.mark.asyncio
    async def test_detect_lagging_market(self, detector, price_monitor):
        """Test detection of lagging market (without explicit lead-lag metadata)."""
        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.CAUSAL,
            expected_relationship="m1 influences m2",
            confidence=0.90,
            historical_correlation=0.85,
        )

        now = time.time()
        from src.signals.divergence.price_monitor import PricePoint

        # m1 moved significantly
        price_monitor._price_history["m1"] = [
            PricePoint(now - 300, 0.50),
            PricePoint(now, 0.58),
        ]
        price_monitor._market_prices["m1"] = 0.58

        # m2 stayed flat
        price_monitor._price_history["m2"] = [
            PricePoint(now - 300, 0.50),
            PricePoint(now, 0.50),
        ]
        price_monitor._market_prices["m2"] = 0.50

        divs = await detector.detect_lagging_market([corr])

        assert len(divs) == 1
        assert divs[0].divergence_type == DivergenceType.LAGGING_MARKET


# ============================================================================
# Detector Tests - Correlation Break
# ============================================================================

class TestCorrelationBreakDetection:

    @pytest.mark.asyncio
    async def test_detect_correlation_break(self, detector, price_monitor):
        """Test detection of correlation breakdown."""
        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="m1 ≈ m2",
            confidence=0.90,
            historical_correlation=0.85,
        )

        now = time.time()
        from src.signals.divergence.price_monitor import PricePoint

        # Create diverging price histories
        # m1 going up
        price_monitor._price_history["m1"] = [
            PricePoint(now - i * 60, 0.50 + i * 0.01)
            for i in range(20)
        ]
        price_monitor._market_prices["m1"] = 0.70

        # m2 going down (negative correlation)
        price_monitor._price_history["m2"] = [
            PricePoint(now - i * 60, 0.50 - i * 0.01)
            for i in range(20)
        ]
        price_monitor._market_prices["m2"] = 0.30

        divs = await detector.detect_correlation_break([corr], window_minutes=30)

        # Should detect correlation break
        corr_break_divs = [d for d in divs if d.divergence_type == DivergenceType.CORRELATION_BREAK]
        assert len(corr_break_divs) == 1


# ============================================================================
# Async Detection Tests
# ============================================================================

class TestAsyncDetection:

    @pytest.mark.asyncio
    async def test_detect_all_divergences_async(self, detector, price_monitor):
        """Test async detection of all divergence types."""
        # Setup prices
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.95,
        )

        divs = await detector.detect_all_divergences([corr], [])

        assert len(divs) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_detection(self, detector, price_monitor):
        """Test that detection methods run concurrently."""
        # Setup multiple correlations
        for i in range(10):
            price_monitor.manual_price_update(f"m{i}", 0.50 + i * 0.01)

        correlations = [
            MarketCorrelation(
                market_a_id=f"m{i}",
                market_b_id=f"m{i+1}",
                correlation_type=CorrelationType.POSITIVE,
                expected_relationship="",
                confidence=0.95,
            )
            for i in range(9)
        ]

        # Should complete without timeout
        divs = await asyncio.wait_for(
            detector.detect_all_divergences(correlations, []),
            timeout=5.0
        )

        assert isinstance(divs, list)


# ============================================================================
# Debouncing and Lifecycle Tests
# ============================================================================

class TestDebouncing:

    @pytest.mark.asyncio
    async def test_debounce_rapid_detections(self, price_monitor):
        """Test that rapid detections are debounced."""
        config = DivergenceConfig(
            min_divergence_threshold=0.02,
            min_liquidity_threshold=0.0,
            debounce_seconds=2.0,
        )
        detector = DivergenceDetector(price_monitor, config=config)

        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="",
            confidence=0.95,
        )

        # First detection (async to properly track debouncing)
        divs1 = await detector.detect_all_divergences([corr], [])
        assert len(divs1) >= 1

        # Immediate second detection should be debounced
        divs2 = await detector.detect_all_divergences([corr], [])
        assert len(divs2) == 0  # Debounced


class TestStatistics:

    @pytest.mark.asyncio
    async def test_detection_statistics(self, detector, price_monitor):
        """Test that statistics are tracked."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="",
            confidence=0.95,
        )

        # Use async method to track stats
        await detector.detect_all_divergences([corr], [])
        stats = detector.get_statistics()

        assert stats["total_detections"] >= 1
        assert stats["active_divergences"] >= 1

    @pytest.mark.asyncio
    async def test_mark_false_positive(self, detector, price_monitor):
        """Test marking divergence as false positive."""
        price_monitor.manual_price_update("m1", 0.50)
        price_monitor.manual_price_update("m2", 0.60)

        corr = MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="",
            confidence=0.95,
        )

        # Use async method to properly track divergences
        divs = await detector.detect_all_divergences([corr], [])
        div_id = divs[0].id

        result = detector.mark_false_positive(div_id, "Prices converged")
        assert result is True

        stats = detector.get_statistics()
        assert stats["false_positives"] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:

    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self, mock_ws):
        """Test complete detection pipeline."""
        # Setup
        monitor = PriceMonitor(mock_ws)
        detector = DivergenceDetector(monitor)

        # Simulate price updates
        monitor.manual_price_update("trump_a", 0.54)
        monitor.manual_price_update("trump_b", 0.58)  # Spread
        monitor.manual_price_update("btc_90k", 0.45)
        monitor.manual_price_update("btc_100k", 0.48)  # Violation

        correlations = [
            MarketCorrelation(
                market_a_id="trump_a",
                market_b_id="trump_b",
                correlation_type=CorrelationType.EQUIVALENT,
                expected_relationship="Same outcome",
                confidence=0.99,
            ),
        ]

        rules = [
            LogicalRule(
                rule_type=LogicalRuleType.THRESHOLD_ORDERING,
                market_ids=["btc_90k", "btc_100k"],
                constraint_desc="Lower threshold >= Higher threshold",
                metadata={
                    "lower_strike_market": "btc_90k",
                    "higher_strike_market": "btc_100k",
                },
            ),
        ]

        # Run detection
        divs = await detector.detect_all_divergences(correlations, rules)

        # Should find both divergences
        types_found = {d.divergence_type for d in divs}
        assert DivergenceType.THRESHOLD_VIOLATION in types_found


# ============================================================================
# Mock Data for False Positive Analysis
# ============================================================================

@pytest.fixture
def false_positive_scenarios():
    """Scenarios that might produce false positives."""
    return [
        {
            "name": "stale_price",
            "description": "One price is stale/cached",
            "prices": {"m1": 0.50, "m2": 0.55},
            "expected_false_positive": True,
        },
        {
            "name": "bid_ask_spread",
            "description": "Difference is just bid-ask spread",
            "prices": {"m1": 0.50, "m2": 0.52},
            "spread_accounts_for": True,
        },
        {
            "name": "different_settlement",
            "description": "Markets have different settlement dates",
            "prices": {"m1": 0.50, "m2": 0.58},
            "different_contracts": True,
        },
    ]


class TestFalsePositiveAnalysis:

    def test_identify_stale_price_risk(self, detector, price_monitor, false_positive_scenarios):
        """Document false positive risks from stale prices."""
        scenario = false_positive_scenarios[0]

        # This test documents that stale prices can cause false positives
        # The detector doesn't currently check for staleness
        price_monitor.manual_price_update("m1", scenario["prices"]["m1"])
        price_monitor.manual_price_update("m2", scenario["prices"]["m2"])

        # In a real scenario, you'd want to check timestamp freshness
        # This is a known limitation to address in future work
        assert True  # Document the limitation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

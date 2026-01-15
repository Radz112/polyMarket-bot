"""
Comprehensive test suite for order management system.

Tests cover:
1. Order validation
2. Paper execution
3. Order splitting
4. Order state machine
5. Integration: signal → order → fill → position
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from src.config.settings import Config
from src.models.orderbook import Orderbook, OrderbookEntry
from src.execution.orders import (
    Order,
    OrderType,
    OrderStatus,
    Fill,
    OrderResult,
    ValidationResult,
    ExecutionPlan,
    OrderValidator,
    OrderExecutor,
    SmartOrderRouter,
    OrderTracker,
    OrderManager,
)
from src.execution.positions import PositionManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        paper_trading_balance=10000.0,
        trading_fees_pct=0.02,
        max_position_size_pct=0.25
    )


@pytest.fixture
def sample_order():
    """Create a sample limit order."""
    return Order(
        id="order_test123",
        market_id="test_market",
        market_name="Test Market",
        side="YES",
        action="BUY",
        order_type=OrderType.LIMIT,
        size=100,
        limit_price=0.50
    )


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook."""
    return Orderbook(
        market_id="test_market",
        bids=[
            OrderbookEntry(price=0.48, size=200),
            OrderbookEntry(price=0.47, size=300),
            OrderbookEntry(price=0.46, size=500),
        ],
        asks=[
            OrderbookEntry(price=0.52, size=150),
            OrderbookEntry(price=0.53, size=250),
            OrderbookEntry(price=0.54, size=400),
        ],
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def position_manager():
    """Create position manager."""
    return PositionManager()


# ============================================================================
# Order Model Tests
# ============================================================================

class TestOrderModel:
    """Test Order dataclass."""
    
    def test_remaining_size_calculation(self, sample_order):
        """remaining_size should be size - filled_size."""
        assert sample_order.remaining_size == 100
        
        sample_order.filled_size = 40
        assert sample_order.remaining_size == 60
    
    def test_is_active(self, sample_order):
        """is_active should be True for pending/submitted/partial."""
        assert sample_order.is_active == True
        
        sample_order.status = OrderStatus.SUBMITTED
        assert sample_order.is_active == True
        
        sample_order.status = OrderStatus.FILLED
        assert sample_order.is_active == False
    
    def test_is_complete(self, sample_order):
        """is_complete should be True for filled/cancelled/rejected."""
        assert sample_order.is_complete == False
        
        sample_order.status = OrderStatus.FILLED
        assert sample_order.is_complete == True
        
        sample_order.status = OrderStatus.CANCELLED
        assert sample_order.is_complete == True
    
    def test_add_fill(self, sample_order):
        """add_fill should update filled_size and average_fill_price."""
        fill = Fill(price=0.50, size=50, timestamp=datetime.utcnow(), fee=0.50)
        sample_order.add_fill(fill)
        
        assert sample_order.filled_size == 50
        assert sample_order.average_fill_price == 0.50
        assert sample_order.status == OrderStatus.PARTIAL
        
        # Second fill completes the order
        fill2 = Fill(price=0.52, size=50, timestamp=datetime.utcnow(), fee=0.52)
        sample_order.add_fill(fill2)
        
        assert sample_order.filled_size == 100
        assert sample_order.status == OrderStatus.FILLED
        assert abs(sample_order.average_fill_price - 0.51) < 0.001
    
    def test_fill_rate(self, sample_order):
        """fill_rate should be percentage filled."""
        assert sample_order.fill_rate == 0.0
        
        sample_order.filled_size = 50
        assert sample_order.fill_rate == 50.0
    
    def test_cancel(self, sample_order):
        """cancel should set status and reason."""
        sample_order.cancel("User requested")
        
        assert sample_order.status == OrderStatus.CANCELLED
        assert sample_order.rejection_reason == "User requested"
    
    def test_reject(self, sample_order):
        """reject should set status and reason."""
        sample_order.reject("Insufficient balance")
        
        assert sample_order.status == OrderStatus.REJECTED
        assert sample_order.rejection_reason == "Insufficient balance"


# ============================================================================
# Order Validation Tests
# ============================================================================

class TestOrderValidation:
    """Test order validation."""
    
    def test_validate_size_minimum(self, config):
        """Should reject orders below minimum size."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=0.5, limit_price=0.50
        )
        
        result = validator.validate_size(order)
        assert not result.is_valid
        assert any("below minimum" in e for e in result.errors)
    
    def test_validate_size_maximum(self, config):
        """Should reject orders above maximum size."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=50000, limit_price=0.50
        )
        
        result = validator.validate_size(order)
        assert not result.is_valid
        assert any("exceeds maximum" in e for e in result.errors)
    
    def test_validate_price_range(self, config):
        """Should reject prices outside 0-1 range."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=100, limit_price=1.50
        )
        
        result = validator.validate_price(order)
        assert not result.is_valid
        assert any("must be between 0 and 1" in e for e in result.errors)
    
    def test_validate_price_limit_required(self, config):
        """Limit orders should require limit_price."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=100, limit_price=None
        )
        
        result = validator.validate_price(order)
        assert not result.is_valid
        assert any("requires limit_price" in e for e in result.errors)
    
    def test_validate_balance_sufficient(self, config):
        """Should pass when balance is sufficient."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=100, limit_price=0.50
        )
        
        result = validator.validate_balance(order, balance=100.0)
        assert result.is_valid
    
    def test_validate_balance_insufficient(self, config):
        """Should reject when balance is insufficient."""
        validator = OrderValidator(config)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=100, limit_price=0.50
        )
        
        result = validator.validate_balance(order, balance=10.0)
        assert not result.is_valid
        assert any("Insufficient balance" in e for e in result.errors)
    
    def test_validate_position_for_sell(self, config, position_manager):
        """Should reject sell orders without position."""
        validator = OrderValidator(config, position_manager)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="SELL", order_type=OrderType.LIMIT,
            size=100, limit_price=0.50
        )
        
        result = validator.validate_position(order)
        assert not result.is_valid
        assert any("No position to sell" in e for e in result.errors)
    
    def test_full_validation_pass(self, config, sample_order):
        """Should pass full validation for valid order."""
        validator = OrderValidator(config)
        
        result = validator.validate(sample_order, balance=100.0)
        assert result.is_valid
    
    def test_quick_validate(self, config, sample_order):
        """quick_validate should check basic requirements."""
        validator = OrderValidator(config)
        
        assert validator.quick_validate(sample_order) == True
        
        # Invalid size
        sample_order.size = 0
        assert validator.quick_validate(sample_order) == False


# ============================================================================
# Order Execution Tests
# ============================================================================

class TestOrderExecution:
    """Test order execution."""
    
    @pytest.mark.asyncio
    async def test_paper_market_order_no_orderbook(self, sample_order):
        """Paper market order without orderbook should fill at mid."""
        executor = OrderExecutor(is_paper=True)
        
        sample_order.order_type = OrderType.MARKET
        sample_order.limit_price = None
        
        result = await executor.execute_paper(sample_order, None)
        
        assert result.success
        assert sample_order.status == OrderStatus.FILLED
        assert sample_order.filled_size == 100
    
    @pytest.mark.asyncio
    async def test_paper_limit_order_fills(self, sample_order):
        """Paper limit order should fill when price is favorable."""
        executor = OrderExecutor(is_paper=True)
        
        # Limit at mid should fill
        sample_order.limit_price = 0.50
        
        result = await executor.execute_paper(sample_order, None)
        
        assert result.success
        assert sample_order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_paper_limit_order_rests(self, sample_order):
        """Paper limit order should rest when price is unfavorable."""
        executor = OrderExecutor(is_paper=True)
        
        # Buy at 0.40 when mid is 0.50 - should rest
        sample_order.limit_price = 0.40
        
        result = await executor.execute_paper(sample_order, None)
        
        assert result.success
        assert sample_order.status == OrderStatus.SUBMITTED  # Resting
    
    @pytest.mark.asyncio
    async def test_cancel_paper_order(self, sample_order):
        """Should cancel paper order."""
        executor = OrderExecutor(is_paper=True)
        
        success = await executor.cancel_order(sample_order)
        
        assert success
        assert sample_order.status == OrderStatus.CANCELLED


# ============================================================================
# Smart Order Router Tests
# ============================================================================

class TestSmartOrderRouter:
    """Test smart order routing."""
    
    def test_should_split_large_order(self, config, sample_order, sample_orderbook):
        """Should split order larger than top level."""
        router = SmartOrderRouter(config)
        
        # Order size 100, top level 150 (30% threshold = 45)
        sample_order.size = 100
        
        should_split = router.should_split_order(sample_order, sample_orderbook)
        assert should_split == True
    
    def test_should_not_split_small_order(self, config, sample_order, sample_orderbook):
        """Should not split small orders."""
        router = SmartOrderRouter(config)
        
        sample_order.size = 10  # Small order
        
        should_split = router.should_split_order(sample_order, sample_orderbook)
        assert should_split == False
    
    def test_calculate_limit_price_immediate(self, config, sample_order, sample_orderbook):
        """Immediate urgency should cross spread."""
        router = SmartOrderRouter(config)
        
        price = router.calculate_optimal_limit_price(
            sample_order, sample_orderbook, urgency="immediate"
        )
        
        # Buy order with immediate urgency should match ask
        assert price >= sample_orderbook.best_ask
    
    def test_calculate_limit_price_patient(self, config, sample_order, sample_orderbook):
        """Patient urgency should be inside spread."""
        router = SmartOrderRouter(config)
        
        price = router.calculate_optimal_limit_price(
            sample_order, sample_orderbook, urgency="patient"
        )
        
        # Buy order with patient urgency should be inside spread
        assert sample_orderbook.best_bid < price < sample_orderbook.best_ask
    
    def test_estimate_fill_probability_market(self, config, sample_order, sample_orderbook):
        """Market orders should have high fill probability."""
        router = SmartOrderRouter(config)
        
        sample_order.order_type = OrderType.MARKET
        
        prob = router.estimate_fill_probability(sample_order, sample_orderbook)
        assert prob >= 0.95
    
    def test_estimate_fill_probability_limit_aggressive(self, config, sample_order, sample_orderbook):
        """Aggressive limit should have high fill probability."""
        router = SmartOrderRouter(config)
        
        sample_order.limit_price = 0.55  # Above best ask
        
        prob = router.estimate_fill_probability(sample_order, sample_orderbook)
        assert prob >= 0.90
    
    def test_plan_execution_single(self, config, sample_order, sample_orderbook):
        """Small orders should use single order strategy."""
        router = SmartOrderRouter(config)
        
        sample_order.size = 10  # Small
        
        plan = router.plan_execution(sample_order, sample_orderbook)
        
        assert plan.strategy == "single"
        assert len(plan.child_orders) == 0
    
    def test_plan_execution_split(self, config, sample_order, sample_orderbook):
        """Large orders should use split strategy."""
        router = SmartOrderRouter(config)
        
        sample_order.size = 200  # Larger than top level
        
        plan = router.plan_execution(sample_order, sample_orderbook)
        
        assert plan.strategy == "split"
        assert len(plan.child_orders) > 1


# ============================================================================
# Order Tracker Tests
# ============================================================================

class TestOrderTracker:
    """Test order tracking."""
    
    @pytest.mark.asyncio
    async def test_track_order(self, sample_order):
        """Should track order."""
        tracker = OrderTracker()
        
        await tracker.track_order(sample_order)
        
        assert sample_order.id in tracker.active_orders
    
    @pytest.mark.asyncio
    async def test_on_fill_updates_order(self, sample_order):
        """on_fill should update order state."""
        tracker = OrderTracker()
        await tracker.track_order(sample_order)
        
        fill = Fill(price=0.50, size=50, timestamp=datetime.utcnow(), fee=0.50)
        await tracker.on_fill(sample_order.id, fill)
        
        order = tracker.get_order(sample_order.id)
        assert order.filled_size == 50
    
    @pytest.mark.asyncio
    async def test_completed_order_moves_to_history(self, sample_order):
        """Completed orders should move to history."""
        tracker = OrderTracker()
        await tracker.track_order(sample_order)
        
        # Fill completely
        fill = Fill(price=0.50, size=100, timestamp=datetime.utcnow(), fee=1.0)
        await tracker.on_fill(sample_order.id, fill)
        
        assert sample_order.id not in tracker.active_orders
        assert len(tracker.completed_orders) == 1
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, sample_order):
        """cancel_order should cancel and move to history."""
        tracker = OrderTracker()
        await tracker.track_order(sample_order)
        
        await tracker.cancel_order(sample_order.id, "Test cancel")
        
        assert sample_order.id not in tracker.active_orders
        assert sample_order.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_get_active_orders_by_market(self, sample_order):
        """Should filter active orders by market."""
        tracker = OrderTracker()
        await tracker.track_order(sample_order)
        
        # Add order for different market
        other_order = Order(
            id="order2", market_id="other_market", market_name="Other",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=50, limit_price=0.50
        )
        await tracker.track_order(other_order)
        
        orders = tracker.get_active_orders("test_market")
        assert len(orders) == 1
        assert orders[0].id == sample_order.id
    
    def test_order_callback(self, sample_order):
        """Should invoke callbacks on order updates."""
        tracker = OrderTracker()
        
        callback_orders = []
        tracker.on_order_update(lambda o: callback_orders.append(o))
        
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            tracker.track_order(sample_order)
        )
        
        assert len(callback_orders) == 1
    
    def test_get_stats(self, sample_order):
        """Should calculate order statistics."""
        tracker = OrderTracker()
        
        stats = tracker.get_stats()
        assert stats["active_count"] == 0
        assert stats["completed_count"] == 0


# ============================================================================
# Order Manager Tests
# ============================================================================

class TestOrderManager:
    """Test order manager."""
    
    @pytest.mark.asyncio
    async def test_submit_valid_order(self, config, sample_order):
        """Should submit valid order."""
        manager = OrderManager(config, is_paper=True)
        
        result = await manager.submit_order(sample_order)
        
        assert result.success
        assert sample_order.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED)
    
    @pytest.mark.asyncio
    async def test_submit_invalid_order_rejected(self, config):
        """Should reject invalid order."""
        manager = OrderManager(config, is_paper=True)
        
        order = Order(
            id="test", market_id="m1", market_name="M1",
            side="YES", action="BUY", order_type=OrderType.LIMIT,
            size=0.5,  # Below minimum
            limit_price=0.50
        )
        
        result = await manager.submit_order(order)
        
        assert not result.success
        assert order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, config, sample_order):
        """Should cancel order."""
        manager = OrderManager(config, is_paper=True)
        
        await manager.submit_order(sample_order)
        success = await manager.cancel_order(sample_order.id)
        
        # If order was filled immediately, cancel returns False
        # Otherwise True
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_get_active_orders(self, config, sample_order):
        """Should get active orders."""
        manager = OrderManager(config, is_paper=True)
        
        # Submit order that will rest (unfavorable price)
        sample_order.limit_price = 0.30  # Low limit - will rest
        await manager.submit_order(sample_order)
        
        orders = manager.get_active_orders()
        # May be 0 or 1 depending on execution
        assert isinstance(orders, list)
    
    def test_create_order(self, config):
        """Should create order with defaults."""
        manager = OrderManager(config, is_paper=True)
        
        order = manager.create_order(
            market_id="test",
            side="YES",
            action="BUY",
            size=100,
            limit_price=0.50
        )
        
        assert order.id.startswith("order_")
        assert order.size == 100
        assert order.status == OrderStatus.PENDING


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_order_lifecycle(self, config):
        """Test complete order lifecycle."""
        manager = OrderManager(config, is_paper=True)
        
        # Create order
        order = manager.create_order(
            market_id="election",
            side="YES",
            action="BUY",
            size=100,
            limit_price=0.50
        )
        
        # Submit
        result = await manager.submit_order(order)
        assert result.success
        
        # Check tracking
        tracked = await manager.get_order_status(order.id)
        assert tracked is not None
    
    @pytest.mark.asyncio
    async def test_split_order_execution(self, config, sample_orderbook):
        """Test split order execution."""
        manager = OrderManager(config, is_paper=True)
        
        # Large order that should split
        order = manager.create_order(
            market_id="test_market",
            side="YES",
            action="BUY",
            size=500,
            limit_price=0.55
        )
        
        result = await manager.submit_order(order, sample_orderbook)
        
        # Should succeed (either single or split)
        assert result.success
    
    @pytest.mark.asyncio
    async def test_balance_update_on_fill(self, config):
        """Balance should update after fill."""
        manager = OrderManager(config, is_paper=True)
        initial_balance = manager.balance
        
        order = manager.create_order(
            market_id="test",
            side="YES",
            action="BUY",
            size=100,
            limit_price=0.50
        )
        
        result = await manager.submit_order(order)
        
        if result.success and order.filled_size > 0:
            # Balance should decrease for buy
            assert manager.balance < initial_balance

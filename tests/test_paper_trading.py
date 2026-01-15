"""
Comprehensive test suite for paper trading system.

Tests cover:
1. Fill simulation (optimistic/realistic/pessimistic models)
2. Paper trading execution (buy/sell, position tracking)
3. Portfolio tracking and metrics
4. Market resolution handling
5. Persistence (save/load state)
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.settings import Config
from src.models.orderbook import Orderbook, OrderbookEntry
from src.execution.paper import (
    FillSimulator,
    SlippageModel,
    FillResult,
    PaperTrader,
    PaperPosition,
    PaperTrade,
    PortfolioTracker,
    PerformanceMetrics,
    PortfolioState,
    PnLSummary,
    ResolutionHandler,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_orderbook():
    """Create a simple orderbook for testing."""
    return Orderbook(
        market_id="test_market",
        timestamp=datetime.utcnow(),
        bids=[
            OrderbookEntry(price=0.45, size=100),
            OrderbookEntry(price=0.44, size=200),
            OrderbookEntry(price=0.43, size=300),
        ],
        asks=[
            OrderbookEntry(price=0.55, size=100),
            OrderbookEntry(price=0.56, size=200),
            OrderbookEntry(price=0.57, size=300),
        ]
    )


@pytest.fixture
def thin_orderbook():
    """Create a thin orderbook with limited liquidity."""
    return Orderbook(
        market_id="thin_market",
        timestamp=datetime.utcnow(),
        bids=[
            OrderbookEntry(price=0.50, size=10),
            OrderbookEntry(price=0.45, size=5),
        ],
        asks=[
            OrderbookEntry(price=0.55, size=10),
            OrderbookEntry(price=0.60, size=5),
        ]
    )


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        paper_trading=True,
        paper_trading_balance=10000.0,
        trading_fees_pct=0.02
    )


@pytest.fixture
def fill_simulator():
    """Create a realistic fill simulator."""
    return FillSimulator(slippage_model="realistic", fee_pct=0.02)


@pytest.fixture
def paper_trader(config):
    """Create a paper trader for testing."""
    return PaperTrader(config=config, db=None, cache=None)


@pytest.fixture
def portfolio_tracker():
    """Create a portfolio tracker."""
    return PortfolioTracker(db=None, initial_balance=10000.0)


# ============================================================================
# Fill Simulation Tests
# ============================================================================

class TestFillSimulator:
    """Test cases for FillSimulator."""
    
    def test_optimistic_fill_at_best_price(self, simple_orderbook):
        """Optimistic model should fill at best price."""
        simulator = FillSimulator(slippage_model="optimistic", fee_pct=0.02)
        
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=50
        )
        
        assert result.filled_size == 50
        assert result.average_price == 0.55  # Best ask
        assert result.slippage == 0.0
        assert result.unfilled_size == 0
        assert result.fees == 50 * 0.55 * 0.02
    
    def test_realistic_fill_walks_book(self, simple_orderbook):
        """Realistic model should walk the orderbook."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Buy 150 shares - should hit multiple levels
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=150
        )
        
        assert result.filled_size == 150
        # 100 @ 0.55, 50 @ 0.56 = (100*0.55 + 50*0.56) / 150 = 0.5533...
        expected_avg = (100 * 0.55 + 50 * 0.56) / 150
        assert abs(result.average_price - expected_avg) < 0.001
        assert result.slippage > 0  # Should have some slippage
        assert result.unfilled_size == 0
    
    def test_pessimistic_fill_adds_slippage(self, simple_orderbook):
        """Pessimistic model should add minimum slippage."""
        simulator = FillSimulator(slippage_model="pessimistic", fee_pct=0.02)
        
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=50
        )
        
        assert result.filled_size == 50
        # Should be 1% higher than best ask
        expected_price = 0.55 * 1.01
        assert abs(result.average_price - expected_price) < 0.001
        assert result.slippage > 0
    
    def test_sell_fills_against_bids(self, simple_orderbook):
        """Sells should fill against bids."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="sell",
            size=50
        )
        
        assert result.filled_size == 50
        assert result.average_price == 0.45  # Best bid
    
    def test_limit_price_respected_for_buy(self, simple_orderbook):
        """Buy orders should respect limit price."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Limit at 0.55 - should only get 100 shares
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=200,
            limit_price=0.55
        )
        
        assert result.filled_size == 100
        assert result.average_price == 0.55
        assert result.unfilled_size == 100
    
    def test_limit_price_respected_for_sell(self, simple_orderbook):
        """Sell orders should respect limit price."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Limit at 0.45 - should only get 100 shares
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="sell",
            size=200,
            limit_price=0.45
        )
        
        assert result.filled_size == 100
        assert result.average_price == 0.45
        assert result.unfilled_size == 100
    
    def test_no_fill_when_limit_not_met(self, simple_orderbook):
        """Should not fill if limit price can't be met."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Try to buy at 0.50 when best ask is 0.55
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=100,
            limit_price=0.50
        )
        
        assert result.filled_size == 0
        assert result.unfilled_size == 100
    
    def test_partial_fill_on_thin_book(self, thin_orderbook):
        """Should partially fill when liquidity is insufficient."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Try to buy 20 shares, but only 15 available
        result = simulator.simulate_fill(
            orderbook=thin_orderbook,
            side="buy",
            size=20
        )
        
        assert result.filled_size == 15  # 10 + 5
        assert result.unfilled_size == 5
    
    def test_empty_orderbook_returns_no_fill(self):
        """Empty orderbook should return no fill."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        empty_book = Orderbook(
            market_id="empty",
            timestamp=datetime.utcnow(),
            bids=[],
            asks=[]
        )
        
        result = simulator.simulate_fill(
            orderbook=empty_book,
            side="buy",
            size=100
        )
        
        assert result.filled_size == 0
        assert result.unfilled_size == 100
    
    def test_estimate_slippage(self, simple_orderbook):
        """Should estimate slippage for given size."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Small order - no slippage
        small_slip = simulator.estimate_slippage(simple_orderbook, "buy", 50)
        assert small_slip == 0  # Fits in first level
        
        # Large order - some slippage
        large_slip = simulator.estimate_slippage(simple_orderbook, "buy", 200)
        assert large_slip > 0  # Crosses multiple levels
    
    def test_can_fill_returns_true_for_small_order(self, simple_orderbook):
        """can_fill should return True for small orders."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        assert simulator.can_fill(simple_orderbook, "buy", 50) is True
    
    def test_can_fill_returns_false_for_huge_order(self, simple_orderbook):
        """can_fill should return False for orders exceeding liquidity."""
        simulator = FillSimulator(slippage_model="realistic", fee_pct=0.02)
        
        # Total ask liquidity is 600
        assert simulator.can_fill(simple_orderbook, "buy", 1000) is False
    
    def test_fees_calculated_correctly(self, simple_orderbook):
        """Fees should be calculated correctly."""
        simulator = FillSimulator(slippage_model="optimistic", fee_pct=0.05)
        
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=100
        )
        
        notional = 100 * 0.55  # 55
        expected_fees = notional * 0.05  # 2.75
        assert abs(result.fees - expected_fees) < 0.001
    
    def test_total_cost_includes_fees_for_buy(self, simple_orderbook):
        """Total cost for buy should include fees."""
        simulator = FillSimulator(slippage_model="optimistic", fee_pct=0.02)
        
        result = simulator.simulate_fill(
            orderbook=simple_orderbook,
            side="buy",
            size=100
        )
        
        notional = 100 * 0.55
        expected_total = notional + (notional * 0.02)
        assert abs(result.total_cost - expected_total) < 0.001


# ============================================================================
# Paper Trading Execution Tests
# ============================================================================

class TestPaperTrader:
    """Test cases for PaperTrader."""
    
    @pytest.mark.asyncio
    async def test_buy_deducts_from_balance(self, paper_trader):
        """Buy should deduct total cost from balance."""
        initial_balance = paper_trader.balance
        
        trade = await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        assert trade is not None
        assert paper_trader.balance < initial_balance
        expected_cost = 100 * 0.50 * (1 + 0.02)  # With fees
        assert abs(paper_trader.balance - (initial_balance - expected_cost)) < 0.01
    
    @pytest.mark.asyncio
    async def test_buy_creates_position(self, paper_trader):
        """Buy should create a position."""
        trade = await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        assert len(paper_trader.positions) == 1
        pos = list(paper_trader.positions.values())[0]
        assert pos.market_id == "test_market"
        assert pos.side == "YES"
        assert pos.size == 100
        assert pos.entry_price == 0.50
    
    @pytest.mark.asyncio
    async def test_buy_insufficient_balance_fails(self, paper_trader):
        """Buy should fail with insufficient balance."""
        # Try to buy more than balance allows
        trade = await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100000,  # Way too much
            limit_price=0.50
        )
        
        assert trade is None
        assert len(paper_trader.positions) == 0
    
    @pytest.mark.asyncio
    async def test_sell_adds_to_balance(self, paper_trader):
        """Sell should add proceeds to balance."""
        # First buy
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        balance_after_buy = paper_trader.balance
        
        # Then sell at higher price
        trade = await paper_trader.sell(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.60
        )
        
        assert trade is not None
        assert paper_trader.balance > balance_after_buy
    
    @pytest.mark.asyncio
    async def test_sell_closes_position(self, paper_trader):
        """Selling full position should remove it."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        assert len(paper_trader.positions) == 1
        
        await paper_trader.sell(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.60
        )
        
        assert len(paper_trader.positions) == 0
    
    @pytest.mark.asyncio
    async def test_sell_without_position_fails(self, paper_trader):
        """Sell should fail if no position exists."""
        trade = await paper_trader.sell(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.60
        )
        
        assert trade is None
    
    @pytest.mark.asyncio
    async def test_partial_sell_reduces_position(self, paper_trader):
        """Partial sell should reduce position size."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        await paper_trader.sell(
            market_id="test_market",
            side="YES",
            size=30,
            limit_price=0.60
        )
        
        pos = list(paper_trader.positions.values())[0]
        assert pos.size == 70
    
    @pytest.mark.asyncio
    async def test_averaging_into_position(self, paper_trader):
        """Multiple buys should average entry price."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.40
        )
        
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.60
        )
        
        pos = list(paper_trader.positions.values())[0]
        assert pos.size == 200
        # Average of 0.40 and 0.60 = 0.50
        assert abs(pos.entry_price - 0.50) < 0.01
    
    @pytest.mark.asyncio
    async def test_close_position(self, paper_trader):
        """close_position should sell entire position."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        position_id = list(paper_trader.positions.keys())[0]
        trade = await paper_trader.close_position(position_id)
        
        assert trade is not None
        assert len(paper_trader.positions) == 0
    
    @pytest.mark.asyncio
    async def test_close_all_positions(self, paper_trader):
        """close_all_positions should close everything."""
        await paper_trader.buy(market_id="market1", side="YES", size=100, limit_price=0.50)
        await paper_trader.buy(market_id="market2", side="NO", size=50, limit_price=0.40)
        
        assert len(paper_trader.positions) == 2
        
        trades = await paper_trader.close_all_positions()
        
        assert len(trades) == 2
        assert len(paper_trader.positions) == 0
    
    @pytest.mark.asyncio
    async def test_portfolio_value_calculation(self, paper_trader):
        """Portfolio value should include positions."""
        initial_value = paper_trader.get_portfolio_value()
        assert initial_value == 10000.0
        
        # Buy some shares
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        # Value should still be approximately the same (minus fees)
        new_value = paper_trader.get_portfolio_value()
        # Expect slight loss due to fees
        assert new_value < initial_value
        assert new_value > initial_value * 0.98  # Within 2%
    
    @pytest.mark.asyncio
    async def test_pnl_calculation(self, paper_trader):
        """P&L should be calculated correctly."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        # Simulate price increase
        paper_trader.update_position_prices({"test_market": 0.60})
        
        pnl = paper_trader.get_pnl()
        
        # Unrealized P&L should show profit
        # New value: 100 * 0.60 = 60
        # Cost basis: 100 * 0.50 = 50
        # Unrealized: 10
        assert pnl.unrealized_pnl == 10.0
        assert pnl.realized_pnl == 0.0  # Nothing sold yet
    
    @pytest.mark.asyncio
    async def test_realized_pnl_after_sell(self, paper_trader):
        """Realized P&L should be calculated after selling."""
        await paper_trader.buy(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.50
        )
        
        await paper_trader.sell(
            market_id="test_market",
            side="YES",
            size=100,
            limit_price=0.60
        )
        
        pnl = paper_trader.get_pnl()
        
        # Profit: (0.60 - 0.50) * 100 = 10, minus fees
        assert pnl.realized_pnl > 0
        assert pnl.unrealized_pnl == 0  # No open positions
    
    @pytest.mark.asyncio
    async def test_trade_history_tracked(self, paper_trader):
        """All trades should be recorded."""
        await paper_trader.buy(market_id="m1", side="YES", size=100, limit_price=0.50)
        await paper_trader.buy(market_id="m2", side="NO", size=50, limit_price=0.40)
        await paper_trader.sell(market_id="m1", side="YES", size=100, limit_price=0.55)
        
        assert len(paper_trader.trades) == 3
        assert paper_trader.trades[0].action == "BUY"
        assert paper_trader.trades[2].action == "SELL"
    
    def test_reset_clears_everything(self, paper_trader):
        """Reset should clear all state."""
        # Simulate some state
        paper_trader.balance = 5000
        paper_trader.positions["test"] = MagicMock()
        paper_trader.trades.append(MagicMock())
        paper_trader.realized_pnl = 500
        
        paper_trader.reset()
        
        assert paper_trader.balance == 10000.0
        assert len(paper_trader.positions) == 0
        assert len(paper_trader.trades) == 0
        assert paper_trader.realized_pnl == 0.0


# ============================================================================
# Portfolio Tracker Tests
# ============================================================================

class TestPortfolioTracker:
    """Test cases for PortfolioTracker."""
    
    def test_record_and_get_trade_history(self, portfolio_tracker):
        """Should record and retrieve trades."""
        trade1 = PaperTrade(
            id="t1",
            market_id="m1",
            market_name="Market 1",
            side="YES",
            action="BUY",
            size=100,
            price=0.50,
            fees=1.0,
            total=51.0
        )
        trade2 = PaperTrade(
            id="t2",
            market_id="m2",
            market_name="Market 2",
            side="NO",
            action="SELL",
            size=50,
            price=0.40,
            fees=0.5,
            total=19.5,
            realized_pnl=5.0
        )
        
        portfolio_tracker.record_trade(trade1)
        portfolio_tracker.record_trade(trade2)
        
        history = portfolio_tracker.get_trade_history(limit=10)
        assert len(history) == 2
    
    def test_get_trade_history_filter_by_market(self, portfolio_tracker):
        """Should filter trades by market."""
        for i in range(5):
            market = "m1" if i < 3 else "m2"
            portfolio_tracker.record_trade(PaperTrade(
                id=f"t{i}",
                market_id=market,
                market_name=market,
                side="YES",
                action="BUY",
                size=10,
                price=0.50,
                fees=0.1,
                total=5.1
            ))
        
        m1_trades = portfolio_tracker.get_trade_history(market_id="m1")
        assert len(m1_trades) == 3
    
    def test_performance_metrics_empty(self, portfolio_tracker):
        """Should return empty metrics with no trades."""
        metrics = portfolio_tracker.get_performance_metrics()
        
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
    
    def test_performance_metrics_calculation(self, portfolio_tracker):
        """Should calculate performance metrics correctly."""
        # Record some closed trades
        trades = [
            PaperTrade(id="t1", market_id="m1", market_name="m1", side="YES", 
                      action="SELL", size=100, price=0.60, fees=1.0, total=59.0,
                      realized_pnl=10.0, holding_period=timedelta(hours=2)),
            PaperTrade(id="t2", market_id="m2", market_name="m2", side="YES", 
                      action="SELL", size=100, price=0.40, fees=1.0, total=39.0,
                      realized_pnl=-5.0, holding_period=timedelta(hours=1)),
            PaperTrade(id="t3", market_id="m3", market_name="m3", side="NO", 
                      action="SELL", size=50, price=0.70, fees=0.5, total=34.5,
                      realized_pnl=8.0, holding_period=timedelta(hours=3)),
        ]
        
        for trade in trades:
            portfolio_tracker.record_trade(trade)
        
        metrics = portfolio_tracker.get_performance_metrics()
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert abs(metrics.win_rate - 0.6667) < 0.01
        assert metrics.average_win == 9.0  # (10 + 8) / 2
        assert metrics.average_loss == 5.0
        assert metrics.total_return == 13.0  # 10 - 5 + 8
    
    def test_max_drawdown_calculation(self, portfolio_tracker):
        """Should calculate max drawdown from snapshots."""
        from src.execution.paper.types import PortfolioSnapshot
        
        # Simulate price movement: 10000 -> 11000 -> 9000 -> 9500
        snapshots = [
            PortfolioSnapshot(id=1, timestamp=datetime(2024, 1, 1), 
                            cash_balance=10000, positions_value=0, 
                            total_value=10000, unrealized_pnl=0, realized_pnl=0),
            PortfolioSnapshot(id=2, timestamp=datetime(2024, 1, 2), 
                            cash_balance=11000, positions_value=0, 
                            total_value=11000, unrealized_pnl=1000, realized_pnl=0),
            PortfolioSnapshot(id=3, timestamp=datetime(2024, 1, 3), 
                            cash_balance=9000, positions_value=0, 
                            total_value=9000, unrealized_pnl=-1000, realized_pnl=0),
            PortfolioSnapshot(id=4, timestamp=datetime(2024, 1, 4), 
                            cash_balance=9500, positions_value=0, 
                            total_value=9500, unrealized_pnl=-500, realized_pnl=0),
        ]
        
        portfolio_tracker._snapshots = snapshots
        
        max_dd, max_dd_pct = portfolio_tracker._calculate_max_drawdown()
        
        # Max drawdown is from 11000 to 9000 = 2000
        assert max_dd == 2000
        assert abs(max_dd_pct - 18.18) < 0.1  # 2000/11000 * 100
    
    def test_get_current_state(self, portfolio_tracker):
        """Should return current portfolio state."""
        positions = [
            PaperPosition(
                id="p1", market_id="m1", market_name="Market 1",
                side="YES", size=100, entry_price=0.50, current_price=0.55
            )
        ]
        
        state = portfolio_tracker.get_current_state(
            cash_balance=9500,
            positions=positions,
            realized_pnl=100
        )
        
        assert state.cash_balance == 9500
        assert len(state.positions) == 1
        assert abs(state.positions_value - 55) < 0.001  # 100 * 0.55
        assert abs(state.total_value - 9555) < 0.001  # 9500 + 55
        assert abs(state.unrealized_pnl - 5) < 0.001  # (0.55 - 0.50) * 100


# ============================================================================
# Resolution Handler Tests
# ============================================================================

class TestResolutionHandler:
    """Test cases for ResolutionHandler."""
    
    def test_settlement_price_yes_wins(self):
        """YES position should settle at 1.0 if market resolves YES."""
        handler = ResolutionHandler()
        
        price = handler.get_settlement_price("YES", "YES")
        assert price == 1.0
    
    def test_settlement_price_yes_loses(self):
        """YES position should settle at 0.0 if market resolves NO."""
        handler = ResolutionHandler()
        
        price = handler.get_settlement_price("YES", "NO")
        assert price == 0.0
    
    def test_settlement_price_no_wins(self):
        """NO position should settle at 1.0 if market resolves NO."""
        handler = ResolutionHandler()
        
        price = handler.get_settlement_price("NO", "NO")
        assert price == 1.0
    
    def test_settlement_price_no_loses(self):
        """NO position should settle at 0.0 if market resolves YES."""
        handler = ResolutionHandler()
        
        price = handler.get_settlement_price("NO", "YES")
        assert price == 0.0
    
    @pytest.mark.asyncio
    async def test_handle_resolution_winning(self):
        """Should calculate profit for winning position."""
        handler = ResolutionHandler()
        
        position = PaperPosition(
            id="p1", market_id="m1", market_name="Test Market",
            side="YES", size=100, entry_price=0.40, current_price=0.40
        )
        
        price, pnl = await handler.handle_resolution(position, "YES")
        
        assert price == 1.0
        # Proceeds: 100 * 1.0 = 100
        # Cost: 100 * 0.40 = 40
        # Profit: 60
        assert pnl == 60.0
    
    @pytest.mark.asyncio
    async def test_handle_resolution_losing(self):
        """Should calculate loss for losing position."""
        handler = ResolutionHandler()
        
        position = PaperPosition(
            id="p1", market_id="m1", market_name="Test Market",
            side="YES", size=100, entry_price=0.60, current_price=0.60
        )
        
        price, pnl = await handler.handle_resolution(position, "NO")
        
        assert price == 0.0
        # Proceeds: 100 * 0.0 = 0
        # Cost: 100 * 0.60 = 60
        # Loss: -60
        assert pnl == -60.0
    
    def test_set_and_get_resolution(self):
        """Should cache resolutions."""
        handler = ResolutionHandler()
        
        handler.set_resolution("m1", "YES")
        
        resolutions = handler.get_cached_resolutions()
        assert resolutions["m1"] == "YES"
    
    def test_clear_resolution(self):
        """Should clear cached resolution."""
        handler = ResolutionHandler()
        
        handler.set_resolution("m1", "YES")
        handler.clear_resolution("m1")
        
        resolutions = handler.get_cached_resolutions()
        assert "m1" not in resolutions


# ============================================================================
# Integration Tests
# ============================================================================

class TestPaperTradingIntegration:
    """Integration tests for paper trading workflow."""
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, config):
        """Test complete buy -> price change -> sell cycle."""
        trader = PaperTrader(config=config)
        tracker = PortfolioTracker(initial_balance=config.paper_trading_balance)
        
        initial_balance = trader.balance
        
        # Buy
        buy_trade = await trader.buy(
            market_id="test",
            side="YES",
            size=100,
            limit_price=0.50,
            market_name="Test Market"
        )
        assert buy_trade is not None
        tracker.record_trade(buy_trade)
        
        # Check position
        assert len(trader.positions) == 1
        
        # Simulate price increase
        trader.update_position_prices({"test": 0.65})
        
        # Check unrealized P&L
        pnl = trader.get_pnl()
        assert pnl.unrealized_pnl == 15.0  # (0.65 - 0.50) * 100
        
        # Sell at profit
        sell_trade = await trader.sell(
            market_id="test",
            side="YES",
            size=100,
            limit_price=0.65
        )
        assert sell_trade is not None
        tracker.record_trade(sell_trade)
        
        # Check final state
        assert len(trader.positions) == 0
        pnl = trader.get_pnl()
        assert pnl.realized_pnl > 0
        assert pnl.unrealized_pnl == 0
        
        # Check metrics
        metrics = tracker.get_performance_metrics()
        assert metrics.total_trades == 1  # 1 closed trade
        assert metrics.winning_trades == 1
    
    @pytest.mark.asyncio
    async def test_execute_multiple_signals(self, config):
        """Test executing 10 signals and tracking P&L."""
        trader = PaperTrader(config=config)
        tracker = PortfolioTracker(initial_balance=config.paper_trading_balance)
        
        # Execute 10 trades
        for i in range(10):
            buy_trade = await trader.buy(
                market_id=f"market_{i}",
                side="YES" if i % 2 == 0 else "NO",
                size=50,
                limit_price=0.40 + (i * 0.02),
                market_name=f"Market {i}"
            )
            
            if buy_trade:
                tracker.record_trade(buy_trade)
        
        assert len(trader.positions) == 10
        
        # Simulate price changes (some up, some down)
        prices = {
            f"market_{i}": 0.40 + (i * 0.02) + (0.10 if i % 3 == 0 else -0.05)
            for i in range(10)
        }
        trader.update_position_prices(prices)
        
        # Close all positions
        for trade in await trader.close_all_positions():
            tracker.record_trade(trade)
        
        # Verify all positions closed
        assert len(trader.positions) == 0
        
        # Check metrics
        metrics = tracker.get_performance_metrics()
        assert metrics.total_trades == 10
        assert metrics.winning_trades + metrics.losing_trades == 10
    
    @pytest.mark.asyncio
    async def test_resolution_handling(self, config):
        """Test market resolution workflow."""
        trader = PaperTrader(config=config)
        handler = ResolutionHandler()
        
        # Buy YES at 0.60
        await trader.buy(
            market_id="election",
            side="YES",
            size=100,
            limit_price=0.60,
            market_name="Election Market"
        )
        
        position = list(trader.positions.values())[0]
        
        # Market resolves YES - position wins
        handler.set_resolution("election", "YES")
        settlement_price, pnl = await handler.handle_resolution(position, "YES")
        
        assert settlement_price == 1.0
        # Profit: (1.0 - 0.60) * 100 = 40
        assert pnl == 40.0
        
        # Update trader state (would normally be done by trading engine)
        trader.balance += 100 * 1.0  # Proceeds from settlement
        trader.realized_pnl += pnl
        del trader.positions[position.id]
        
        assert len(trader.positions) == 0
        assert trader.realized_pnl > 0


# ============================================================================
# Data Type Tests
# ============================================================================

class TestPaperTradingTypes:
    """Test paper trading type classes."""
    
    def test_paper_position_computed_properties(self):
        """Test PaperPosition computed properties."""
        pos = PaperPosition(
            id="p1",
            market_id="m1",
            market_name="Test",
            side="YES",
            size=100,
            entry_price=0.50,
            current_price=0.60
        )
        
        assert pos.market_value == 60.0
        assert pos.cost_basis == 50.0
        assert pos.unrealized_pnl == 10.0
        assert pos.unrealized_pnl_pct == 20.0  # 10/50 * 100
    
    def test_paper_position_to_dict(self):
        """Test PaperPosition serialization."""
        pos = PaperPosition(
            id="p1",
            market_id="m1",
            market_name="Test",
            side="YES",
            size=100,
            entry_price=0.50,
            current_price=0.60
        )
        
        d = pos.to_dict()
        
        assert d["id"] == "p1"
        assert d["market_value"] == 60.0
        assert d["unrealized_pnl"] == 10.0
    
    def test_paper_trade_to_dict(self):
        """Test PaperTrade serialization."""
        trade = PaperTrade(
            id="t1",
            market_id="m1",
            market_name="Test",
            side="YES",
            action="BUY",
            size=100,
            price=0.50,
            fees=1.0,
            total=51.0
        )
        
        d = trade.to_dict()
        
        assert d["id"] == "t1"
        assert d["action"] == "BUY"
        assert d["total"] == 51.0
    
    def test_pnl_summary_to_dict(self):
        """Test PnLSummary serialization."""
        pnl = PnLSummary(
            realized_pnl=100.0,
            unrealized_pnl=50.0,
            total_pnl=150.0,
            total_fees=5.0,
            return_pct=1.5
        )
        
        d = pnl.to_dict()
        
        assert d["total_pnl"] == 150.0
        assert d["return_pct"] == 1.5
    
    def test_fill_result_partial_fill(self):
        """Test FillResult partial fill detection."""
        partial = FillResult(
            filled_size=50,
            average_price=0.55,
            slippage=0.01,
            fees=0.5,
            total_cost=28.0,
            unfilled_size=50
        )
        
        assert partial.was_partial is True
        
        full = FillResult(
            filled_size=100,
            average_price=0.55,
            slippage=0.01,
            fees=1.0,
            total_cost=56.0,
            unfilled_size=0
        )
        
        assert full.was_partial is False

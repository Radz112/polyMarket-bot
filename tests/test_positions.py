"""
Comprehensive test suite for position tracking system.

Tests cover:
1. Position P&L calculations
2. Exposure tracking with correlations
3. Alert triggering
4. Position history
5. Performance with many positions
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from src.config.settings import Config
from src.execution.positions import (
    Position,
    PositionStatus,
    ClosedPosition,
    PositionAlert,
    AlertType,
    AlertSeverity,
    ConcentrationReport,
    PositionAnalytics,
    PositionManager,
    ExposureTracker,
    PositionAlerts,
    PositionHistory,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        position_pnl_alert_pct=0.10,
        position_time_alert_hours=24,
        position_resolution_warning_hours=24
    )


@pytest.fixture
def position_manager():
    """Create position manager."""
    return PositionManager(db=None, cache=None)


@pytest.fixture
def sample_position():
    """Create a sample position."""
    return Position(
        id="pos_test_YES_abc123",
        market_id="test_market",
        market_name="Test Market",
        category="Politics",
        side="YES",
        size=100,
        entry_price=0.50,
        entry_time=datetime.utcnow() - timedelta(hours=2),
        current_price=0.55
    )


@pytest.fixture
def multiple_positions():
    """Create multiple positions for testing."""
    return [
        Position(
            id=f"pos_{i}",
            market_id=f"market_{i}",
            market_name=f"Market {i}",
            category="Politics" if i < 3 else "Crypto",
            side="YES" if i % 2 == 0 else "NO",
            size=100,
            entry_price=0.40 + i * 0.05,
            entry_time=datetime.utcnow() - timedelta(hours=i),
            current_price=0.45 + i * 0.05
        )
        for i in range(5)
    ]


# ============================================================================
# Position P&L Tests
# ============================================================================

class TestPositionPnL:
    """Test position P&L calculations."""
    
    def test_cost_basis_calculation(self, sample_position):
        """Cost basis should be size * entry price."""
        assert sample_position.cost_basis == 100 * 0.50
        assert sample_position.cost_basis == 50.0
    
    def test_market_value_calculation(self, sample_position):
        """Market value should be size * current price."""
        assert abs(sample_position.market_value - 55.0) < 0.001
    
    def test_unrealized_pnl_positive(self, sample_position):
        """Unrealized P&L should be positive when price increased."""
        assert abs(sample_position.unrealized_pnl - 5.0) < 0.001  # 55 - 50
    
    def test_unrealized_pnl_negative(self, sample_position):
        """Unrealized P&L should be negative when price decreased."""
        sample_position.current_price = 0.45
        assert sample_position.unrealized_pnl == -5.0  # 45 - 50
    
    def test_unrealized_pnl_pct(self, sample_position):
        """P&L percentage should be calculated correctly."""
        # 5/50 * 100 = 10%
        assert abs(sample_position.unrealized_pnl_pct - 10.0) < 0.001
    
    def test_update_price(self, sample_position):
        """update_price should update all price fields."""
        sample_position.update_price(0.60, 0.58, 0.62)
        
        assert sample_position.current_price == 0.60
        assert sample_position.current_bid == 0.58
        assert sample_position.current_ask == 0.62
        assert abs(sample_position.spread - 0.04) < 0.001
    
    def test_time_held(self, sample_position):
        """time_held should return correct duration."""
        # Position was opened 2 hours ago
        assert sample_position.time_held.total_seconds() >= 2 * 3600 - 1
        assert sample_position.time_held.total_seconds() < 3 * 3600


# ============================================================================
# Position Manager Tests
# ============================================================================

class TestPositionManager:
    """Test PositionManager operations."""
    
    @pytest.mark.asyncio
    async def test_open_new_position(self, position_manager):
        """Should open a new position."""
        position = await position_manager.open_position(
            market_id="test_market",
            side="YES",
            size=100,
            entry_price=0.50,
            market_name="Test Market",
            category="Politics"
        )
        
        assert position is not None
        assert position.size == 100
        assert position.entry_price == 0.50
        assert len(position_manager.positions) == 1
    
    @pytest.mark.asyncio
    async def test_average_into_position(self, position_manager):
        """Adding to position should average entry price."""
        # First buy at 0.50
        await position_manager.open_position(
            market_id="test_market",
            side="YES",
            size=100,
            entry_price=0.50
        )
        
        # Second buy at 0.60
        position = await position_manager.open_position(
            market_id="test_market",
            side="YES",
            size=100,
            entry_price=0.60
        )
        
        # Should have 1 position with averaged entry
        assert len(position_manager.positions) == 1
        assert position.size == 200
        assert position.entry_price == 0.55  # Average of 0.50 and 0.60
    
    @pytest.mark.asyncio
    async def test_reduce_position(self, position_manager):
        """Reducing position should calculate P&L."""
        await position_manager.open_position(
            market_id="test_market",
            side="YES",
            size=100,
            entry_price=0.50
        )
        
        position = position_manager.get_position("test_market", "YES")
        updated, pnl = await position_manager.reduce_position(
            position.id,
            size=50,
            exit_price=0.60
        )
        
        assert updated.size == 50
        assert abs(pnl - 5.0) < 0.001  # (0.60 - 0.50) * 50
    
    @pytest.mark.asyncio
    async def test_close_position(self, position_manager):
        """Closing position should calculate P&L and remove it."""
        await position_manager.open_position(
            market_id="test_market",
            side="YES",
            size=100,
            entry_price=0.50
        )
        
        position = position_manager.get_position("test_market", "YES")
        pnl = await position_manager.close_position(position.id, exit_price=0.70)
        
        assert abs(pnl - 20.0) < 0.001  # (0.70 - 0.50) * 100
        assert len(position_manager.positions) == 0
        assert len(position_manager.closed_positions) == 1
    
    def test_get_position_by_market_and_side(self, position_manager, sample_position):
        """Should find position by market and side."""
        position_manager._add_position(sample_position)
        
        found = position_manager.get_position("test_market", "YES")
        assert found is not None
        assert found.id == sample_position.id
        
        not_found = position_manager.get_position("test_market", "NO")
        assert not_found is None
    
    def test_get_positions_by_market(self, position_manager, multiple_positions):
        """Should get all positions in a market."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        market_0_positions = position_manager.get_positions_by_market("market_0")
        assert len(market_0_positions) == 1
    
    def test_get_positions_by_category(self, position_manager, multiple_positions):
        """Should get all positions in a category."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        politics_positions = position_manager.get_positions_by_category("Politics")
        assert len(politics_positions) == 3
        
        crypto_positions = position_manager.get_positions_by_category("Crypto")
        assert len(crypto_positions) == 2
    
    def test_get_total_exposure(self, position_manager, multiple_positions):
        """Should calculate total exposure."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        total = position_manager.get_total_exposure()
        expected = sum(p.market_value for p in multiple_positions)
        assert abs(total - expected) < 0.01
    
    def test_get_exposure_by_category(self, position_manager, multiple_positions):
        """Should group exposure by category."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        by_category = position_manager.get_exposure_by_category()
        
        assert "Politics" in by_category
        assert "Crypto" in by_category


# ============================================================================
# Exposure Tracker Tests
# ============================================================================

class TestExposureTracker:
    """Test exposure tracking."""
    
    def test_get_category_exposure(self, position_manager, multiple_positions):
        """Should calculate category exposure."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        tracker = ExposureTracker(position_manager)
        by_category = tracker.get_category_exposure()
        
        assert len(by_category) == 2
    
    def test_get_exposure_summary(self, position_manager, multiple_positions):
        """Should generate exposure summary."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        tracker = ExposureTracker(position_manager)
        summary = tracker.get_exposure_summary()
        
        assert summary.total_exposure > 0
        assert summary.largest_single_exposure > 0
    
    def test_get_concentration_report(self, position_manager, multiple_positions):
        """Should generate concentration report."""
        for pos in multiple_positions:
            position_manager._add_position(pos)
        
        tracker = ExposureTracker(position_manager)
        report = tracker.get_concentration_report()
        
        assert report.largest_position is not None
        assert report.largest_position_pct > 0
        assert 0 <= report.diversification_score <= 100
    
    def test_concentration_report_empty(self, position_manager):
        """Empty portfolio should have high diversification."""
        tracker = ExposureTracker(position_manager)
        report = tracker.get_concentration_report()
        
        assert report.diversification_score == 100
        assert report.largest_position is None
    
    def test_correlated_exposure(self, position_manager):
        """Should calculate correlated exposure."""
        pos1 = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Trump PA",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.55,
            correlated_positions=["pos_2"]
        )
        pos2 = Position(
            id="pos_2",
            market_id="market_2",
            market_name="Trump National",
            category="Politics",
            side="YES",
            size=50,
            entry_price=0.60,
            entry_time=datetime.utcnow(),
            current_price=0.65
        )
        
        position_manager._add_position(pos1)
        position_manager._add_position(pos2)
        
        tracker = ExposureTracker(position_manager)
        corr_exposure = tracker.get_correlated_exposure(pos1)
        
        # Should include both positions' values
        assert corr_exposure > pos1.market_value


# ============================================================================
# Position Alerts Tests
# ============================================================================

class TestPositionAlerts:
    """Test alert generation."""
    
    @pytest.mark.asyncio
    async def test_pnl_gain_alert(self, position_manager, config):
        """Should alert on large gains."""
        # Create position with 15% gain (above 10% threshold)
        pos = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Test",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.575  # 15% gain
        )
        position_manager._add_position(pos)
        
        alerts_checker = PositionAlerts(position_manager, config)
        alerts = await alerts_checker.check_alerts()
        
        pnl_alerts = [a for a in alerts if a.alert_type == AlertType.PNL_GAIN]
        assert len(pnl_alerts) == 1
        assert pnl_alerts[0].severity == AlertSeverity.INFO
    
    @pytest.mark.asyncio
    async def test_pnl_loss_alert(self, position_manager, config):
        """Should alert on large losses."""
        # Create position with 15% loss
        pos = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Test",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.425  # 15% loss
        )
        position_manager._add_position(pos)
        
        alerts_checker = PositionAlerts(position_manager, config)
        alerts = await alerts_checker.check_alerts()
        
        pnl_alerts = [a for a in alerts if a.alert_type == AlertType.PNL_LOSS]
        assert len(pnl_alerts) == 1
        assert pnl_alerts[0].severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_time_held_alert(self, position_manager, config):
        """Should alert when position held too long."""
        # Create position opened 30 hours ago
        pos = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Test",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow() - timedelta(hours=30),
            current_price=0.50
        )
        position_manager._add_position(pos)
        
        alerts_checker = PositionAlerts(position_manager, config)
        alerts = await alerts_checker.check_alerts()
        
        time_alerts = [a for a in alerts if a.alert_type == AlertType.TIME_HELD]
        assert len(time_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_resolution_near_alert(self, position_manager, config):
        """Should alert when resolution is near."""
        pos = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Test",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.50,
            distance_to_resolution=timedelta(hours=12)  # Within 24h threshold
        )
        position_manager._add_position(pos)
        
        alerts_checker = PositionAlerts(position_manager, config)
        alerts = await alerts_checker.check_alerts()
        
        res_alerts = [a for a in alerts if a.alert_type == AlertType.RESOLUTION_NEAR]
        assert len(res_alerts) == 1
        assert res_alerts[0].severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_correlation_conflict_alert(self, position_manager, config):
        """Should alert on correlated position conflicts."""
        pos1 = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Trump PA",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.50,
            correlated_positions=["pos_2"]
        )
        pos2 = Position(
            id="pos_2",
            market_id="market_2",
            market_name="Trump National",
            category="Politics",
            side="NO",  # Opposite side - conflict
            size=50,
            entry_price=0.50,
            entry_time=datetime.utcnow(),
            current_price=0.50
        )
        
        position_manager._add_position(pos1)
        position_manager._add_position(pos2)
        
        alerts_checker = PositionAlerts(position_manager, config)
        alerts = await alerts_checker.check_alerts()
        
        corr_alerts = [a for a in alerts if a.alert_type == AlertType.CORRELATION_CONFLICT]
        assert len(corr_alerts) == 1
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, position_manager, config):
        """Should not repeat alerts."""
        pos = Position(
            id="pos_1",
            market_id="market_1",
            market_name="Test",
            category="Politics",
            side="YES",
            size=100,
            entry_price=0.50,
            entry_time=datetime.utcnow() - timedelta(hours=30),
            current_price=0.50
        )
        position_manager._add_position(pos)
        
        alerts_checker = PositionAlerts(position_manager, config)
        
        # First check
        alerts1 = await alerts_checker.check_alerts()
        assert len(alerts1) > 0
        
        # Second check - should be empty (deduplicated)
        alerts2 = await alerts_checker.check_alerts()
        assert len(alerts2) == 0


# ============================================================================
# Position History Tests
# ============================================================================

class TestPositionHistory:
    """Test position history tracking."""
    
    def test_record_snapshot(self, sample_position):
        """Should record position snapshot."""
        history = PositionHistory()
        history.record_snapshot(sample_position)
        
        snapshots = history.get_position_history(sample_position.id)
        assert len(snapshots) == 1
        assert snapshots[0].price == sample_position.current_price
    
    def test_record_multiple_snapshots(self, sample_position):
        """Should record multiple snapshots over time."""
        history = PositionHistory()
        
        for i in range(5):
            sample_position.current_price = 0.50 + i * 0.02
            history.record_snapshot(sample_position)
        
        snapshots = history.get_position_history(sample_position.id)
        assert len(snapshots) == 5
    
    def test_record_closed_position(self, sample_position):
        """Should record closed positions."""
        history = PositionHistory()
        
        closed = ClosedPosition(
            position=sample_position,
            exit_price=0.60,
            exit_time=datetime.utcnow(),
            realized_pnl=10.0,
            holding_period=timedelta(hours=2),
            exit_reason="manual"
        )
        history.record_closed_position(closed)
        
        closed_positions = history.get_closed_positions()
        assert len(closed_positions) == 1
        assert closed_positions[0].realized_pnl == 10.0
    
    def test_get_closed_by_date_range(self, sample_position):
        """Should filter closed positions by date."""
        history = PositionHistory()
        
        # Add positions closed at different times
        now = datetime.utcnow()
        for i in range(5):
            closed = ClosedPosition(
                position=sample_position,
                exit_price=0.60,
                exit_time=now - timedelta(days=i),
                realized_pnl=10.0,
                holding_period=timedelta(hours=2),
                exit_reason="manual"
            )
            history.record_closed_position(closed)
        
        # Get last 2 days
        start = now - timedelta(days=2)
        filtered = history.get_closed_positions(start=start)
        assert len(filtered) == 3
    
    def test_position_analytics(self, sample_position):
        """Should generate position analytics."""
        history = PositionHistory()
        
        # Record price movement
        prices = [0.50, 0.55, 0.60, 0.58, 0.52]
        for price in prices:
            sample_position.current_price = price
            history.record_snapshot(sample_position)
        
        analytics = history.get_position_analytics(sample_position.id)
        
        assert analytics is not None
        assert analytics.max_gain > 0
        assert analytics.snapshots_count == 5
    
    def test_summary_stats(self, sample_position):
        """Should calculate summary statistics."""
        history = PositionHistory()
        
        # Add closed positions
        for i in range(10):
            closed = ClosedPosition(
                position=sample_position,
                exit_price=0.55 + (i % 3) * 0.05,
                exit_time=datetime.utcnow(),
                realized_pnl=5.0 if i % 3 != 0 else -3.0,
                holding_period=timedelta(hours=2 + i),
                exit_reason="manual"
            )
            history.record_closed_position(closed)
        
        stats = history.get_summary_stats()
        
        assert stats["total_closed"] == 10
        assert "win_rate" in stats
        assert "avg_holding_period_hours" in stats


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance with many positions."""
    
    def test_50_positions_lookup_speed(self, position_manager):
        """Lookups should be fast with 50+ positions."""
        import time
        
        # Add 50 positions
        for i in range(50):
            pos = Position(
                id=f"pos_{i}",
                market_id=f"market_{i}",
                market_name=f"Market {i}",
                category=f"Category_{i % 5}",
                side="YES" if i % 2 == 0 else "NO",
                size=100,
                entry_price=0.50,
                entry_time=datetime.utcnow(),
                current_price=0.55
            )
            position_manager._add_position(pos)
        
        # Test lookup speed
        start = time.perf_counter()
        for i in range(50):
            position_manager.get_position(f"market_{i}", "YES" if i % 2 == 0 else "NO")
        elapsed = time.perf_counter() - start
        
        # Should complete in < 10ms
        assert elapsed < 0.010, f"Lookups took {elapsed*1000:.2f}ms"
    
    def test_exposure_calculation_speed(self, position_manager):
        """Exposure calculations should be fast."""
        import time
        
        # Add 50 positions
        for i in range(50):
            pos = Position(
                id=f"pos_{i}",
                market_id=f"market_{i}",
                market_name=f"Market {i}",
                category=f"Category_{i % 5}",
                side="YES",
                size=100,
                entry_price=0.50,
                entry_time=datetime.utcnow(),
                current_price=0.55
            )
            position_manager._add_position(pos)
        
        tracker = ExposureTracker(position_manager)
        
        # Test calculation speed
        start = time.perf_counter()
        tracker.get_exposure_summary()
        tracker.get_concentration_report()
        elapsed = time.perf_counter() - start
        
        # Should complete in < 10ms
        assert elapsed < 0.010, f"Calculations took {elapsed*1000:.2f}ms"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for position tracking."""
    
    @pytest.mark.asyncio
    async def test_full_position_lifecycle(self, config):
        """Test complete position lifecycle."""
        manager = PositionManager()
        history = PositionHistory()
        alerts = PositionAlerts(manager, config)
        
        # Open position
        position = await manager.open_position(
            market_id="election",
            side="YES",
            size=100,
            entry_price=0.50,
            market_name="2024 Election",
            category="Politics"
        )
        
        # Record initial snapshot
        history.record_snapshot(position)
        
        # Price updates
        for price in [0.52, 0.55, 0.58, 0.60]:
            manager.update_position_price("election", price)
            history.record_snapshot(manager.get_position("election", "YES"))
        
        # Check for alerts
        alerts_list = await alerts.check_alerts()
        
        # Close position
        pnl = await manager.close_position(position.id, exit_price=0.60)
        
        # Record closed position
        history.record_closed_position(manager.closed_positions[-1])
        
        # Verify
        assert abs(pnl - 10.0) < 0.001  # (0.60 - 0.50) * 100
        assert len(history.get_position_history(position.id)) == 5
        assert len(history.get_closed_positions()) == 1

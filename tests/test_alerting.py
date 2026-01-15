"""
Tests for the alerting system.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from src.alerting.formatters import (
    Alert, AlertType, AlertSeverity, AlertChannel,
    DailySummary, AlertFormatter
)
from src.alerting.rate_limiter import AlertRateLimiter, RateLimitConfig
from src.alerting.rules import AlertRule, AlertRulesEngine
from src.alerting.telegram import TelegramClient, TelegramConfig
from src.alerting.email import EmailClient, EmailConfig
from src.alerting.manager import AlertManager, AlertManagerConfig


# ==================== Test Fixtures ====================

@dataclass
class MockSignal:
    id: str = "sig_123"
    score: int = 85
    signal_type: str = "divergence"
    divergence_amount: float = 0.05
    recommended_action: str = "BUY"
    recommended_size: float = 200
    markets: list = None
    
    def __post_init__(self):
        if self.markets is None:
            self.markets = [
                {"id": "m1", "question": "Will Trump win?", "yesPrice": 0.52},
                {"id": "m2", "question": "Will Biden win?", "yesPrice": 0.48},
            ]


@dataclass
class MockTrade:
    id: str = "trade_123"
    market_name: str = "Will Trump win the 2024 election?"
    action: str = "BUY"
    side: str = "YES"
    size: float = 150
    price: float = 0.52
    fees: float = 3.0
    realized_pnl: float = None


@dataclass
class MockPosition:
    id: str = "pos_123"
    market_name: str = "Will Trump win?"
    side: str = "YES"
    size: float = 200
    entry_price: float = 0.48
    current_price: float = 0.52
    unrealized_pnl: float = 16.67
    unrealized_pnl_pct: float = 8.33


@dataclass
class MockRiskEvent:
    alert_type: str = "Daily Loss Warning"
    severity: str = "warning"
    message: str = "Daily loss at 5% of limit"
    daily_loss_pct: float = 5
    breaker_tripped: bool = False
    trading_halted: bool = False


# ==================== Rate Limiter Tests ====================

class TestAlertRateLimiter:
    """Tests for AlertRateLimiter."""
    
    def test_should_send_within_limits(self):
        """Test that alerts are allowed within limits."""
        limiter = AlertRateLimiter(RateLimitConfig(max_per_hour=10, max_per_rule_per_hour=3))
        
        assert limiter.should_send("test_rule") is True
    
    def test_global_limit_blocks(self):
        """Test that global limit blocks alerts."""
        limiter = AlertRateLimiter(RateLimitConfig(max_per_hour=3, max_per_rule_per_hour=10))
        
        # Send 3 alerts
        for _ in range(3):
            limiter.record_alert("rule1")
        
        # 4th should be blocked
        assert limiter.should_send("different_rule") is False
    
    def test_per_rule_limit_blocks(self):
        """Test that per-rule limit blocks alerts for that rule."""
        limiter = AlertRateLimiter(RateLimitConfig(max_per_hour=100, max_per_rule_per_hour=2))
        
        # Send 2 alerts for same rule
        limiter.record_alert("test_rule")
        limiter.record_alert("test_rule")
        
        # Same rule should be blocked
        assert limiter.should_send("test_rule") is False
        
        # Different rule should be allowed
        assert limiter.should_send("other_rule") is True
    
    def test_cooldown_blocks(self):
        """Test that cooldown blocks alerts."""
        limiter = AlertRateLimiter(RateLimitConfig(cooldown_seconds=60))
        
        limiter.record_alert("test_rule")
        
        # Should be blocked due to cooldown
        assert limiter.should_send("test_rule", timedelta(seconds=60)) is False
    
    def test_cooldown_override(self):
        """Test cooldown override."""
        limiter = AlertRateLimiter(RateLimitConfig(cooldown_seconds=3600))
        
        limiter.record_alert("test_rule")
        
        # With short cooldown override, should still be blocked
        assert limiter.should_send("test_rule", timedelta(seconds=1)) is False
    
    def test_get_stats(self):
        """Test getting statistics."""
        limiter = AlertRateLimiter(RateLimitConfig(max_per_hour=10))
        
        limiter.record_alert("rule1")
        limiter.record_alert("rule1")
        limiter.record_alert("rule2")
        
        stats = limiter.get_stats()
        
        assert stats["alerts_last_hour"] == 3
        assert stats["max_per_hour"] == 10
        assert stats["remaining"] == 7
        assert stats["by_rule"]["rule1"] == 2
        assert stats["by_rule"]["rule2"] == 1
    
    def test_reset(self):
        """Test reset functionality."""
        limiter = AlertRateLimiter()
        
        limiter.record_alert("rule1")
        limiter.record_alert("rule2")
        
        limiter.reset()
        
        stats = limiter.get_stats()
        assert stats["alerts_last_hour"] == 0


# ==================== Alert Rules Tests ====================

class TestAlertRulesEngine:
    """Tests for AlertRulesEngine."""
    
    def test_default_rules_created(self):
        """Test that default rules are created."""
        engine = AlertRulesEngine()
        
        assert len(engine.rules) > 0
        
        rule_names = [r.name for r in engine.rules]
        assert "high_score_signal" in rule_names
        assert "circuit_breaker_tripped" in rule_names
    
    def test_evaluate_signal_rule(self):
        """Test signal rule evaluation."""
        engine = AlertRulesEngine()
        
        signal = MockSignal(score=85)
        context = {"signal": signal}
        
        matched = engine.evaluate("signal", context)
        
        # Should match high_score_signal (>= 80)
        rule_names = [r.name for r in matched]
        assert "high_score_signal" in rule_names
    
    def test_evaluate_low_score_signal(self):
        """Test that low score signal doesn't match high_score rule."""
        engine = AlertRulesEngine()
        
        signal = MockSignal(score=50)
        context = {"signal": signal}
        
        matched = engine.evaluate("signal", context)
        
        # Should not match high_score_signal
        rule_names = [r.name for r in matched]
        assert "high_score_signal" not in rule_names
    
    def test_evaluate_risk_rules(self):
        """Test risk rule evaluation."""
        engine = AlertRulesEngine()
        
        context = {
            "daily_loss_pct": 6,
            "breaker_tripped": False,
            "trading_halted": False,
        }
        
        matched = engine.evaluate("risk", context)
        
        # Should match daily_loss_warning (>= 5%)
        rule_names = [r.name for r in matched]
        assert "daily_loss_warning" in rule_names
    
    def test_circuit_breaker_rule(self):
        """Test circuit breaker rule."""
        engine = AlertRulesEngine()
        
        context = {"breaker_tripped": True}
        
        matched = engine.evaluate("risk", context)
        
        rule_names = [r.name for r in matched]
        assert "circuit_breaker_tripped" in rule_names
    
    def test_enable_disable_rule(self):
        """Test enabling and disabling rules."""
        engine = AlertRulesEngine()
        
        # Disable rule
        engine.enable_rule("high_score_signal", False)
        
        signal = MockSignal(score=85)
        context = {"signal": signal}
        
        matched = engine.evaluate("signal", context)
        
        # Should not match disabled rule
        rule_names = [r.name for r in matched]
        assert "high_score_signal" not in rule_names
        
        # Re-enable
        engine.enable_rule("high_score_signal", True)
        
        matched = engine.evaluate("signal", context)
        rule_names = [r.name for r in matched]
        assert "high_score_signal" in rule_names
    
    def test_add_custom_rule(self):
        """Test adding custom rules."""
        engine = AlertRulesEngine()
        
        custom_rule = AlertRule(
            name="custom_test",
            description="Test rule",
            event_type="signal",
            condition=lambda ctx: ctx.get('signal') and ctx['signal'].score > 50,
            channels=[AlertChannel.TELEGRAM],
            severity=AlertSeverity.INFO,
        )
        
        engine.add_rule(custom_rule)
        
        assert "custom_test" in [r.name for r in engine.rules]
    
    def test_get_rule_config(self):
        """Test getting rule configuration."""
        engine = AlertRulesEngine()
        
        config = engine.get_rule_config()
        
        assert len(config) > 0
        assert all("name" in c for c in config)
        assert all("enabled" in c for c in config)


# ==================== Formatter Tests ====================

class TestAlertFormatter:
    """Tests for AlertFormatter."""
    
    def test_format_signal_telegram(self):
        """Test signal formatting for Telegram."""
        signal = MockSignal()
        
        text = AlertFormatter.format_signal_telegram(signal)
        
        assert "NEW SIGNAL" in text
        assert "Score: 85" in text
        assert "Will Trump win?" in text
        assert "5.0¢" in text  # divergence
        assert "BUY" in text
    
    def test_format_trade_telegram(self):
        """Test trade formatting for Telegram."""
        trade = MockTrade()
        
        text = AlertFormatter.format_trade_telegram(trade)
        
        assert "TRADE EXECUTED" in text
        assert "Trump" in text
        assert "BUY YES" in text
        assert "$150.00" in text
    
    def test_format_trade_with_pnl(self):
        """Test trade formatting with P&L."""
        trade = MockTrade(realized_pnl=25.50)
        
        text = AlertFormatter.format_trade_telegram(trade)
        
        assert "$+25.50" in text
        assert "📈" in text
    
    def test_format_risk_telegram(self):
        """Test risk alert formatting."""
        event = MockRiskEvent()
        
        text = AlertFormatter.format_risk_telegram(event)
        
        assert "RISK ALERT" in text
        assert "Daily Loss Warning" in text
        assert "WARNING" in text
    
    def test_format_position_telegram(self):
        """Test position alert formatting."""
        position = MockPosition()
        
        text = AlertFormatter.format_position_telegram(position, "Large gain")
        
        assert "POSITION UPDATE" in text
        assert "Large gain" in text
        assert "$+16.67" in text
        assert "+8.3%" in text
    
    def test_format_daily_summary(self):
        """Test daily summary formatting."""
        summary = DailySummary(
            date=datetime.now(),
            total_pnl=127.50,
            total_pnl_pct=2.1,
            trades_count=12,
            winning_trades=9,
            losing_trades=3,
            win_rate=0.75,
            open_positions=5,
            total_position_value=1500,
            unrealized_pnl=45.20,
            top_trades=[
                {"market": "Trump wins", "pnl": 45},
                {"market": "BTC > 100k", "pnl": 32},
            ],
            risk_metrics={"daily_loss_used_pct": 15, "max_drawdown_pct": 3.2},
        )
        
        text = AlertFormatter.format_daily_summary_telegram(summary)
        
        assert "DAILY SUMMARY" in text
        assert "$+127.50" in text
        assert "75%" in text
        assert "Trump wins" in text
    
    def test_format_daily_email_html(self):
        """Test daily email HTML formatting."""
        summary = DailySummary(
            date=datetime.now(),
            total_pnl=127.50,
            total_pnl_pct=2.1,
            trades_count=12,
            winning_trades=9,
            losing_trades=3,
            win_rate=0.75,
            open_positions=5,
            total_position_value=1500,
            unrealized_pnl=45.20,
            top_trades=[{"market": "Test", "pnl": 50}],
            risk_metrics={},
        )
        
        html = AlertFormatter.format_daily_email_html(summary)
        
        assert "<html>" in html
        assert "Daily Report" in html
        assert "$+127.50" in html


# ==================== Alert Manager Tests ====================

class TestAlertManager:
    """Tests for AlertManager."""
    
    def test_initialization(self):
        """Test AlertManager initialization."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        assert manager.rate_limiter is not None
        assert manager.rules_engine is not None
    
    @pytest.mark.asyncio
    async def test_send_alert_to_telegram(self):
        """Test sending alert to Telegram."""
        config = AlertManagerConfig(
            telegram=TelegramConfig(
                bot_token="test_token",
                chat_id="test_chat",
                enabled=True
            )
        )
        manager = AlertManager(config)
        manager.telegram = MagicMock()
        manager.telegram.is_available = True
        manager.telegram.send_message = AsyncMock(return_value=True)
        
        alert = Alert(
            id="alert_1",
            type=AlertType.SIGNAL,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            channels=[AlertChannel.TELEGRAM],
            rule_name="test_rule",
        )
        
        result = await manager.send_alert(alert)
        
        assert result is True
        manager.telegram.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_alert_rate_limited(self):
        """Test that rate-limited alerts are not sent."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        # Fill up rate limit
        manager.rate_limiter = MagicMock()
        manager.rate_limiter.should_send.return_value = False
        
        alert = Alert(
            id="alert_1",
            type=AlertType.SIGNAL,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            channels=[AlertChannel.TELEGRAM],
            rule_name="test_rule",
        )
        
        result = await manager.send_alert(alert)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_on_signal_high_score(self):
        """Test handling high score signal."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        manager.telegram = MagicMock()
        manager.telegram.is_available = True
        manager.telegram.send_message = AsyncMock(return_value=True)
        
        signal = MockSignal(score=85)
        
        await manager.on_signal(signal)
        
        # Should have sent alert
        assert manager.telegram.send_message.called
    
    @pytest.mark.asyncio  
    async def test_on_signal_low_score(self):
        """Test that low score signals don't trigger alerts."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        manager.telegram = MagicMock()
        manager.telegram.is_available = True
        manager.telegram.send_message = AsyncMock(return_value=True)
        
        signal = MockSignal(score=50)
        
        await manager.on_signal(signal)
        
        # Should not have sent alert
        assert not manager.telegram.send_message.called
    
    def test_get_stats(self):
        """Test getting statistics."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        stats = manager.get_stats()
        
        assert "rate_limiter" in stats
        assert "alerts_sent" in stats
        assert "rules_count" in stats
    
    def test_update_rule(self):
        """Test updating rule state."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        # Disable a rule
        manager.update_rule("high_score_signal", False)
        
        # Check it's disabled
        rule = next(r for r in manager.rules_engine.rules if r.name == "high_score_signal")
        assert rule.enabled is False


# ==================== Integration Tests ====================

class TestAlertingIntegration:
    """Integration tests for the alerting system."""
    
    @pytest.mark.asyncio
    async def test_full_signal_flow(self):
        """Test full flow from signal to alert."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        # Mock telegram
        manager.telegram = MagicMock()
        manager.telegram.is_available = True
        manager.telegram.send_message = AsyncMock(return_value=True)
        
        # High score signal
        signal = MockSignal(score=90)
        
        await manager.on_signal(signal)
        
        # Verify alert was sent
        assert manager.telegram.send_message.called
        
        # Verify rate limit recorded
        stats = manager.rate_limiter.get_stats()
        assert stats["alerts_last_hour"] > 0
    
    @pytest.mark.asyncio
    async def test_risk_event_to_email(self):
        """Test risk event triggering email."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        # Mock both channels
        manager.telegram = MagicMock()
        manager.telegram.is_available = True
        manager.telegram.send_message = AsyncMock(return_value=True)
        
        manager.email = MagicMock()
        manager.email.is_available = True
        manager.email.send_risk_alert = AsyncMock(return_value=True)
        
        # Critical risk event
        event = MockRiskEvent()
        event.severity = "critical"
        event.daily_loss_pct = 9
        
        await manager.on_risk_event(event)
        
        # Should send to both channels for critical
        # Note: email is sent separately for high severity
        assert manager.email.send_risk_alert.called

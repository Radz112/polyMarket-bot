"""
Alert Manager - orchestrates alert sending across channels.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from dataclasses import dataclass

from .formatters import Alert, AlertType, AlertSeverity, AlertChannel, DailySummary
from .rate_limiter import AlertRateLimiter, RateLimitConfig
from .rules import AlertRulesEngine, AlertRule
from .telegram import TelegramClient, TelegramConfig
from .email import EmailClient, EmailConfig

logger = logging.getLogger(__name__)


@dataclass
class AlertManagerConfig:
    """Configuration for AlertManager."""
    telegram: Optional[TelegramConfig] = None
    email: Optional[EmailConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    log_alerts: bool = True


class AlertManager:
    """
    Central manager for all alerts.
    
    Handles:
    - Evaluating alert rules
    - Rate limiting
    - Sending to appropriate channels
    - Logging and tracking
    """
    
    def __init__(self, config: AlertManagerConfig):
        self.config = config
        
        # Initialize clients
        self.telegram = TelegramClient(config.telegram) if config.telegram else None
        self.email = EmailClient(config.email) if config.email else None
        
        # Initialize components
        self.rate_limiter = AlertRateLimiter(config.rate_limit)
        self.rules_engine = AlertRulesEngine()
        
        # Tracking
        self.sent_alerts: list = []
        self._running = False
        
        logger.info("AlertManager initialized")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent to at least one channel
        """
        # Check rate limiting
        if alert.rule_name and not self.rate_limiter.should_send(alert.rule_name):
            logger.debug(f"Alert rate-limited: {alert.rule_name}")
            return False
        
        sent_to_any = False
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.TELEGRAM and self.telegram:
                    success = await self.telegram.send_message(alert.message)
                    if success:
                        sent_to_any = True
                        logger.debug(f"Alert sent to Telegram: {alert.title}")
                
                elif channel == AlertChannel.EMAIL and self.email:
                    success = await self.email.send_email(
                        subject=f"{alert.severity_emoji} {alert.title}",
                        body_text=alert.message,
                    )
                    if success:
                        sent_to_any = True
                        logger.debug(f"Alert sent to Email: {alert.title}")
                        
            except Exception as e:
                logger.error(f"Error sending alert to {channel.value}: {e}")
        
        # Record if sent
        if sent_to_any and alert.rule_name:
            self.rate_limiter.record_alert(alert.rule_name)
            
            if self.config.log_alerts:
                self.sent_alerts.append({
                    "id": alert.id,
                    "type": alert.type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "sent_at": datetime.utcnow().isoformat(),
                    "channels": [c.value for c in alert.channels],
                })
        
        return sent_to_any
    
    async def on_signal(self, signal: Any):
        """
        Handle new signal event.
        
        Args:
            signal: The signal that was detected
        """
        context = {"signal": signal}
        matched_rules = self.rules_engine.evaluate("signal", context)
        
        for rule in matched_rules:
            if not self.rate_limiter.should_send(rule.name, rule.cooldown):
                continue
            
            from .formatters import AlertFormatter
            message = AlertFormatter.format_signal_telegram(signal)
            
            alert = Alert(
                id=str(uuid.uuid4()),
                type=AlertType.SIGNAL,
                severity=rule.severity,
                title=f"New Signal (Score: {signal.score})",
                message=message,
                channels=rule.channels,
                data={"signal_id": signal.id},
                rule_name=rule.name,
            )
            
            await self.send_alert(alert)
    
    async def on_trade(self, trade: Any):
        """
        Handle trade execution event.
        
        Args:
            trade: The executed trade
        """
        context = {"trade": trade}
        matched_rules = self.rules_engine.evaluate("trade", context)
        
        for rule in matched_rules:
            if not self.rate_limiter.should_send(rule.name, rule.cooldown):
                continue
            
            from .formatters import AlertFormatter
            message = AlertFormatter.format_trade_telegram(trade)
            
            alert = Alert(
                id=str(uuid.uuid4()),
                type=AlertType.TRADE,
                severity=rule.severity,
                title=f"Trade Executed: {trade.action} {trade.side}",
                message=message,
                channels=rule.channels,
                data={"trade_id": trade.id},
                rule_name=rule.name,
            )
            
            await self.send_alert(alert)
    
    async def on_position_update(self, position: Any):
        """
        Handle position update event.
        
        Args:
            position: The updated position
        """
        context = {"position": position}
        matched_rules = self.rules_engine.evaluate("position", context)
        
        for rule in matched_rules:
            if not self.rate_limiter.should_send(rule.name, rule.cooldown):
                continue
            
            reason = "Large gain" if position.unrealized_pnl_pct >= 0 else "Large loss"
            
            from .formatters import AlertFormatter
            message = AlertFormatter.format_position_telegram(position, reason)
            
            alert = Alert(
                id=str(uuid.uuid4()),
                type=AlertType.POSITION,
                severity=rule.severity,
                title=f"Position Alert: {reason}",
                message=message,
                channels=rule.channels,
                data={"position_id": position.id},
                rule_name=rule.name,
            )
            
            await self.send_alert(alert)
    
    async def on_risk_event(self, event: Any):
        """
        Handle risk event.
        
        Args:
            event: The risk event
        """
        context = {
            "daily_loss_pct": getattr(event, 'daily_loss_pct', 0),
            "breaker_tripped": getattr(event, 'breaker_tripped', False),
            "trading_halted": getattr(event, 'trading_halted', False),
        }
        matched_rules = self.rules_engine.evaluate("risk", context)
        
        for rule in matched_rules:
            # Critical risk alerts bypass cooldown
            if rule.severity != AlertSeverity.CRITICAL:
                if not self.rate_limiter.should_send(rule.name, rule.cooldown):
                    continue
            
            from .formatters import AlertFormatter
            message = AlertFormatter.format_risk_telegram(event)
            
            alert = Alert(
                id=str(uuid.uuid4()),
                type=AlertType.RISK,
                severity=rule.severity,
                title=f"Risk Alert: {getattr(event, 'alert_type', 'Unknown')}",
                message=message,
                channels=rule.channels,
                data={"event_type": getattr(event, 'alert_type', 'unknown')},
                rule_name=rule.name,
            )
            
            await self.send_alert(alert)
            
            # Also send email for high severity
            if rule.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] and self.email:
                await self.email.send_risk_alert(event)
    
    async def send_daily_summary(self, summary: DailySummary):
        """
        Send daily summary to all channels.
        
        Args:
            summary: The daily summary data
        """
        # Send to Telegram
        if self.telegram and self.telegram.is_available:
            await self.telegram.send_daily_summary(summary)
        
        # Send email report
        if self.email and self.email.is_available:
            await self.email.send_daily_report(summary)
        
        logger.info("Daily summary sent")
    
    def get_stats(self) -> dict:
        """Get alerting statistics."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "alerts_sent": len(self.sent_alerts),
            "recent_alerts": self.sent_alerts[-10:],
            "telegram_available": self.telegram.is_available if self.telegram else False,
            "email_available": self.email.is_available if self.email else False,
            "rules_count": len(self.rules_engine.rules),
        }
    
    def get_rule_config(self) -> list:
        """Get current rule configuration."""
        return self.rules_engine.get_rule_config()
    
    def update_rule(self, rule_name: str, enabled: bool):
        """Enable or disable a rule."""
        self.rules_engine.enable_rule(rule_name, enabled)
    
    async def test_channels(self) -> dict:
        """Test all configured channels."""
        results = {}
        
        if self.telegram:
            results["telegram"] = await self.telegram.test_connection()
        else:
            results["telegram"] = {"success": False, "error": "Not configured"}
        
        if self.email:
            results["email"] = await self.email.test_connection()
        else:
            results["email"] = {"success": False, "error": "Not configured"}
        
        return results

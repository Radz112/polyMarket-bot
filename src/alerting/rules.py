"""
Alert rules engine for evaluating when to send alerts.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, List, Dict, Any, Optional
from enum import Enum
import logging

from .formatters import Alert, AlertType, AlertSeverity, AlertChannel

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    event_type: str  # signal, trade, position, risk
    condition: Callable[[dict], bool]
    channels: List[AlertChannel]
    severity: AlertSeverity
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    enabled: bool = True
    
    def evaluate(self, context: dict) -> bool:
        """Evaluate if rule condition is met."""
        if not self.enabled:
            return False
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(f"Error evaluating rule '{self.name}': {e}")
            return False


class AlertRulesEngine:
    """
    Engine for evaluating alert rules.
    
    Manages a set of rules and evaluates them against incoming events.
    """
    
    def __init__(self):
        self.rules: List[AlertRule] = self._setup_default_rules()
    
    def _setup_default_rules(self) -> List[AlertRule]:
        """Set up default alert rules."""
        return [
            # Signal alerts
            AlertRule(
                name="high_score_signal",
                description="Alert when a high-score signal is detected",
                event_type="signal",
                condition=lambda ctx: ctx.get('signal') and ctx['signal'].score >= 80,
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.HIGH,
                cooldown=timedelta(minutes=5),
            ),
            AlertRule(
                name="critical_score_signal",
                description="Alert for very high score signals",
                event_type="signal",
                condition=lambda ctx: ctx.get('signal') and ctx['signal'].score >= 90,
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.CRITICAL,
                cooldown=timedelta(minutes=1),
            ),
            
            # Trade alerts
            AlertRule(
                name="trade_executed",
                description="Alert when a trade is executed",
                event_type="trade",
                condition=lambda ctx: ctx.get('trade') is not None,
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.INFO,
                cooldown=timedelta(seconds=30),
            ),
            AlertRule(
                name="large_trade",
                description="Alert for large trades",
                event_type="trade",
                condition=lambda ctx: ctx.get('trade') and ctx['trade'].size >= 500,
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.HIGH,
                cooldown=timedelta(minutes=1),
            ),
            
            # Position alerts
            AlertRule(
                name="large_position_gain",
                description="Alert when position has significant gain",
                event_type="position",
                condition=lambda ctx: (
                    ctx.get('position') and 
                    ctx['position'].unrealized_pnl_pct >= 20
                ),
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.INFO,
                cooldown=timedelta(hours=2),
            ),
            AlertRule(
                name="large_position_loss",
                description="Alert when position has significant loss",
                event_type="position",
                condition=lambda ctx: (
                    ctx.get('position') and 
                    ctx['position'].unrealized_pnl_pct <= -15
                ),
                channels=[AlertChannel.TELEGRAM],
                severity=AlertSeverity.WARNING,
                cooldown=timedelta(hours=1),
            ),
            
            # Risk alerts
            AlertRule(
                name="daily_loss_warning",
                description="Warning when daily loss approaches limit",
                event_type="risk",
                condition=lambda ctx: ctx.get('daily_loss_pct', 0) >= 5,
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                severity=AlertSeverity.WARNING,
                cooldown=timedelta(hours=1),
            ),
            AlertRule(
                name="daily_loss_critical",
                description="Critical alert when daily loss is high",
                event_type="risk",
                condition=lambda ctx: ctx.get('daily_loss_pct', 0) >= 8,
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                severity=AlertSeverity.CRITICAL,
                cooldown=timedelta(minutes=15),
            ),
            AlertRule(
                name="circuit_breaker_tripped",
                description="Alert when circuit breaker trips",
                event_type="risk",
                condition=lambda ctx: ctx.get('breaker_tripped', False),
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                severity=AlertSeverity.CRITICAL,
                cooldown=timedelta(seconds=0),  # Always alert
            ),
            AlertRule(
                name="trading_halted",
                description="Alert when trading is halted",
                event_type="risk",
                condition=lambda ctx: ctx.get('trading_halted', False),
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL],
                severity=AlertSeverity.CRITICAL,
                cooldown=timedelta(seconds=0),
            ),
        ]
    
    def add_rule(self, rule: AlertRule):
        """Add a new rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str, enabled: bool = True):
        """Enable or disable a rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                logger.info(f"Rule '{rule_name}' {'enabled' if enabled else 'disabled'}")
                return
    
    def evaluate(self, event_type: str, context: dict) -> List[AlertRule]:
        """
        Evaluate all rules for a given event type.
        
        Args:
            event_type: Type of event (signal, trade, position, risk)
            context: Event context data
            
        Returns:
            List of rules that matched
        """
        matched_rules = []
        
        for rule in self.rules:
            if rule.event_type != event_type:
                continue
            
            if rule.evaluate(context):
                matched_rules.append(rule)
                logger.debug(f"Rule matched: {rule.name}")
        
        return matched_rules
    
    def get_rules(self, event_type: Optional[str] = None) -> List[AlertRule]:
        """Get all rules, optionally filtered by event type."""
        if event_type:
            return [r for r in self.rules if r.event_type == event_type]
        return self.rules.copy()
    
    def get_rule_config(self) -> List[dict]:
        """Get rule configuration for settings UI."""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "event_type": rule.event_type,
                "channels": [c.value for c in rule.channels],
                "severity": rule.severity.value,
                "cooldown_seconds": int(rule.cooldown.total_seconds()),
                "enabled": rule.enabled,
            }
            for rule in self.rules
        ]

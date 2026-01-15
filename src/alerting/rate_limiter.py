"""
Alert rate limiter to prevent spam.
"""
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_per_hour: int = 30
    max_per_rule_per_hour: int = 5
    cooldown_seconds: int = 60


class AlertRateLimiter:
    """
    Rate limiter for alerts to prevent spam.
    
    Features:
    - Global hourly limit
    - Per-rule hourly limit  
    - Cooldown between same rule alerts
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.recent_alerts: deque = deque()
        self.alerts_by_rule: Dict[str, deque] = defaultdict(deque)
        self.last_alert_by_rule: Dict[str, datetime] = {}
    
    def should_send(self, rule_name: str, cooldown_override: Optional[timedelta] = None) -> bool:
        """
        Check if alert should be sent based on rate limits.
        
        Args:
            rule_name: Name of the alert rule
            cooldown_override: Optional cooldown override for this specific check
            
        Returns:
            True if alert should be sent, False otherwise
        """
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self._cleanup_old_entries(hour_ago)
        
        # Check global limit
        if len(self.recent_alerts) >= self.config.max_per_hour:
            logger.debug(f"Rate limit hit: global limit ({self.config.max_per_hour}/hour)")
            return False
        
        # Check per-rule limit
        rule_alerts = self.alerts_by_rule[rule_name]
        if len(rule_alerts) >= self.config.max_per_rule_per_hour:
            logger.debug(f"Rate limit hit: rule '{rule_name}' limit ({self.config.max_per_rule_per_hour}/hour)")
            return False
        
        # Check cooldown
        cooldown = cooldown_override or timedelta(seconds=self.config.cooldown_seconds)
        last_alert = self.last_alert_by_rule.get(rule_name)
        if last_alert and (now - last_alert) < cooldown:
            remaining = (last_alert + cooldown - now).total_seconds()
            logger.debug(f"Rate limit hit: rule '{rule_name}' cooldown ({remaining:.0f}s remaining)")
            return False
        
        return True
    
    def record_alert(self, rule_name: str):
        """
        Record that an alert was sent.
        
        Args:
            rule_name: Name of the alert rule
        """
        now = datetime.utcnow()
        self.recent_alerts.append(now)
        self.alerts_by_rule[rule_name].append(now)
        self.last_alert_by_rule[rule_name] = now
        logger.debug(f"Recorded alert for rule '{rule_name}'")
    
    def _cleanup_old_entries(self, cutoff: datetime):
        """Remove entries older than cutoff time."""
        # Clean global alerts
        while self.recent_alerts and self.recent_alerts[0] < cutoff:
            self.recent_alerts.popleft()
        
        # Clean per-rule alerts
        for rule_name, alerts in self.alerts_by_rule.items():
            while alerts and alerts[0] < cutoff:
                alerts.popleft()
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        self._cleanup_old_entries(hour_ago)
        
        return {
            "alerts_last_hour": len(self.recent_alerts),
            "max_per_hour": self.config.max_per_hour,
            "remaining": max(0, self.config.max_per_hour - len(self.recent_alerts)),
            "by_rule": {
                rule: len(alerts) 
                for rule, alerts in self.alerts_by_rule.items()
                if len(alerts) > 0
            }
        }
    
    def reset(self):
        """Reset all rate limit counters."""
        self.recent_alerts.clear()
        self.alerts_by_rule.clear()
        self.last_alert_by_rule.clear()
        logger.info("Rate limiter reset")

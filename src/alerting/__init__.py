"""
Alerting system for Polymarket Bot.

Provides Telegram and Email alerts for signals, trades, and risk events.
"""
from .formatters import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertChannel,
    DailySummary,
    AlertFormatter,
)
from .rate_limiter import AlertRateLimiter, RateLimitConfig
from .rules import AlertRule, AlertRulesEngine
from .telegram import TelegramClient, TelegramConfig, TelegramCommandHandler
from .email import EmailClient, EmailConfig
from .manager import AlertManager, AlertManagerConfig

__all__ = [
    # Models
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertChannel",
    "DailySummary",
    "AlertFormatter",
    
    # Rate limiting
    "AlertRateLimiter",
    "RateLimitConfig",
    
    # Rules
    "AlertRule",
    "AlertRulesEngine",
    
    # Telegram
    "TelegramClient",
    "TelegramConfig",
    "TelegramCommandHandler",
    
    # Email
    "EmailClient",
    "EmailConfig",
    
    # Manager
    "AlertManager",
    "AlertManagerConfig",
]

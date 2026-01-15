"""
Alert models and formatters for the alerting system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Any


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Available alert channels."""
    TELEGRAM = "telegram"
    EMAIL = "email"


class AlertType(Enum):
    """Types of alerts."""
    SIGNAL = "signal"
    TRADE = "trade"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Alert:
    """Represents an alert to be sent."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    channels: List[AlertChannel]
    data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    rule_name: Optional[str] = None
    
    @property
    def severity_emoji(self) -> str:
        """Get emoji for severity level."""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "🔴",
        }.get(self.severity, "📢")


@dataclass
class DailySummary:
    """Daily performance summary for reports."""
    date: datetime
    total_pnl: float
    total_pnl_pct: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    open_positions: int
    total_position_value: float
    unrealized_pnl: float
    top_trades: List[dict]
    risk_metrics: dict


class AlertFormatter:
    """Format alerts for different channels."""
    
    @staticmethod
    def format_signal_telegram(signal: Any) -> str:
        """Format signal alert for Telegram."""
        markets_text = ""
        for market in signal.markets[:2]:
            price = market.get('yesPrice', 0) * 100
            markets_text += f"• {market.get('question', 'Unknown')[:50]}: {price:.1f}¢\n"
        
        return f"""🚨 <b>NEW SIGNAL</b> (Score: {signal.score})

<b>Type:</b> {signal.signal_type.value if hasattr(signal.signal_type, 'value') else signal.signal_type}

<b>Markets:</b>
{markets_text}
<b>Divergence:</b> {signal.divergence_amount * 100:.1f}¢
<b>Action:</b> {signal.recommended_action}
<b>Size:</b> ${signal.recommended_size:.0f}

⏰ <i>Act quickly - window may close</i>"""
    
    @staticmethod
    def format_trade_telegram(trade: Any) -> str:
        """Format trade confirmation for Telegram."""
        pnl_text = ""
        if trade.realized_pnl is not None:
            pnl_emoji = "📈" if trade.realized_pnl >= 0 else "📉"
            pnl_text = f"\n<b>P&L:</b> {pnl_emoji} ${trade.realized_pnl:+.2f}"
        
        return f"""✅ <b>TRADE EXECUTED</b>

<b>Market:</b> {trade.market_name[:50]}
<b>Action:</b> {trade.action} {trade.side}
<b>Size:</b> ${trade.size:.2f}
<b>Price:</b> {trade.price * 100:.1f}¢
<b>Fees:</b> ${trade.fees:.2f}{pnl_text}"""
    
    @staticmethod
    def format_risk_telegram(event: Any) -> str:
        """Format risk alert for Telegram."""
        severity_color = {
            "warning": "🟡",
            "high": "🟠", 
            "critical": "🔴",
        }.get(event.severity, "⚪")
        
        return f"""⚠️ <b>RISK ALERT</b>

<b>Type:</b> {event.alert_type}
<b>Severity:</b> {severity_color} {event.severity.upper()}

{event.message}

<i>Use /status for current state</i>"""
    
    @staticmethod
    def format_position_telegram(position: Any, alert_reason: str) -> str:
        """Format position alert for Telegram."""
        pnl_emoji = "📈" if position.unrealized_pnl >= 0 else "📉"
        
        return f"""📊 <b>POSITION UPDATE</b>

<b>Reason:</b> {alert_reason}

<b>Market:</b> {position.market_name[:50]}
<b>Side:</b> {position.side}
<b>Size:</b> ${position.size:.2f}
<b>Entry:</b> {position.entry_price * 100:.1f}¢
<b>Current:</b> {position.current_price * 100:.1f}¢

<b>P&L:</b> {pnl_emoji} ${position.unrealized_pnl:+.2f} ({position.unrealized_pnl_pct:+.1f}%)"""
    
    @staticmethod
    def format_daily_summary_telegram(summary: DailySummary) -> str:
        """Format daily summary for Telegram."""
        pnl_emoji = "📈" if summary.total_pnl >= 0 else "📉"
        
        top_trades_text = ""
        for i, trade in enumerate(summary.top_trades[:3], 1):
            top_trades_text += f"{i}. {trade['market'][:30]}: ${trade['pnl']:+.2f}\n"
        
        return f"""📊 <b>DAILY SUMMARY</b> - {summary.date.strftime('%b %d, %Y')}

<b>Performance:</b>
{pnl_emoji} P&L: ${summary.total_pnl:+.2f} ({summary.total_pnl_pct:+.1f}%)
Trades: {summary.trades_count} (Win rate: {summary.win_rate*100:.0f}%)

<b>Top Trades:</b>
{top_trades_text}
<b>Open Positions:</b> {summary.open_positions}
Value: ${summary.total_position_value:.2f}
Unrealized: ${summary.unrealized_pnl:+.2f}

<b>Risk Status:</b>
Daily Loss Used: {summary.risk_metrics.get('daily_loss_used_pct', 0):.1f}%
Max Drawdown: {summary.risk_metrics.get('max_drawdown_pct', 0):.1f}%"""
    
    @staticmethod
    def format_daily_email_html(summary: DailySummary) -> str:
        """Format daily summary as HTML email."""
        pnl_color = "#22c55e" if summary.total_pnl >= 0 else "#ef4444"
        
        top_trades_html = ""
        for trade in summary.top_trades[:5]:
            color = "#22c55e" if trade['pnl'] >= 0 else "#ef4444"
            top_trades_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;">{trade['market'][:40]}</td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; color: {color}; font-weight: bold;">${trade['pnl']:+.2f}</td>
            </tr>"""
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 30px; color: white; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .metric-row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .metric {{ flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric .value {{ font-size: 28px; font-weight: bold; color: #1a1a1a; }}
        .metric .label {{ font-size: 12px; color: #666; text-transform: uppercase; margin-top: 5px; }}
        .section {{ margin-top: 30px; }}
        .section h2 {{ margin: 0 0 15px 0; font-size: 16px; color: #666; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Polymarket Bot</h1>
            <p>Daily Report - {summary.date.strftime('%B %d, %Y')}</p>
        </div>
        <div class="content">
            <div class="metric-row">
                <div class="metric">
                    <div class="value" style="color: {pnl_color}">${summary.total_pnl:+.2f}</div>
                    <div class="label">Daily P&L</div>
                </div>
                <div class="metric">
                    <div class="value">{summary.trades_count}</div>
                    <div class="label">Trades</div>
                </div>
                <div class="metric">
                    <div class="value">{summary.win_rate*100:.0f}%</div>
                    <div class="label">Win Rate</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top Trades</h2>
                <table>
                    {top_trades_html}
                </table>
            </div>
            
            <div class="section">
                <h2>Open Positions</h2>
                <p>You have <strong>{summary.open_positions}</strong> open positions worth <strong>${summary.total_position_value:.2f}</strong></p>
                <p>Unrealized P&L: <span style="color: {'#22c55e' if summary.unrealized_pnl >= 0 else '#ef4444'}">${summary.unrealized_pnl:+.2f}</span></p>
            </div>
            
            <div class="section">
                <h2>Risk Status</h2>
                <p>Daily Loss Used: <strong>{summary.risk_metrics.get('daily_loss_used_pct', 0):.1f}%</strong> of limit</p>
                <p>Max Drawdown: <strong>{summary.risk_metrics.get('max_drawdown_pct', 0):.1f}%</strong></p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    @staticmethod
    def format_daily_email_text(summary: DailySummary) -> str:
        """Format daily summary as plain text email."""
        top_trades_text = "\n".join(
            f"  {i}. {t['market'][:40]}: ${t['pnl']:+.2f}"
            for i, t in enumerate(summary.top_trades[:5], 1)
        )
        
        return f"""
POLYMARKET BOT - DAILY REPORT
{summary.date.strftime('%B %d, %Y')}
{'='*40}

PERFORMANCE SUMMARY
-------------------
Daily P&L: ${summary.total_pnl:+.2f} ({summary.total_pnl_pct:+.1f}%)
Trades: {summary.trades_count}
Winning: {summary.winning_trades}
Losing: {summary.losing_trades}
Win Rate: {summary.win_rate*100:.0f}%

TOP TRADES
----------
{top_trades_text}

OPEN POSITIONS
--------------
Count: {summary.open_positions}
Total Value: ${summary.total_position_value:.2f}
Unrealized P&L: ${summary.unrealized_pnl:+.2f}

RISK STATUS
-----------
Daily Loss Used: {summary.risk_metrics.get('daily_loss_used_pct', 0):.1f}% of limit
Max Drawdown: {summary.risk_metrics.get('max_drawdown_pct', 0):.1f}%
Circuit Breakers: {'OK' if not summary.risk_metrics.get('breakers_tripped') else 'TRIPPED'}
"""

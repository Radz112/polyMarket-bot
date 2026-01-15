"""
Signal formatting for various output formats.

Supports dashboard display, Telegram alerts, CSV export, and JSON API responses.
"""
import json
import csv
import io
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from src.signals.scoring.types import ScoredSignal, RecommendedAction, Urgency
from src.signals.filtering.ranker import RankedSignal, SignalTier

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Supported output formats."""
    DASHBOARD = "dashboard"
    TELEGRAM = "telegram"
    CSV = "csv"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    SLACK = "slack"


@dataclass
class FormatterConfig:
    """Configuration for signal formatting."""
    # Dashboard settings
    dashboard_show_details: bool = True
    dashboard_max_signals: int = 20

    # Telegram settings
    telegram_emoji_enabled: bool = True
    telegram_max_length: int = 4000  # Telegram message limit
    telegram_include_links: bool = True

    # CSV settings
    csv_delimiter: str = ","
    csv_include_header: bool = True

    # JSON settings
    json_pretty: bool = False
    json_include_metadata: bool = True

    # General
    include_timestamps: bool = True
    timezone: str = "UTC"
    decimal_places: int = 2


class SignalFormatter:
    """
    Formats signals for various output channels.

    Supports:
    - Dashboard (rich HTML/terminal display)
    - Telegram (mobile-friendly with emoji)
    - CSV (for data export)
    - JSON (for API responses)
    - Plain text (for logs)
    - Slack (for Slack webhooks)
    """

    def __init__(self, config: FormatterConfig = None):
        self.config = config or FormatterConfig()

    def format(
        self,
        signals: List[RankedSignal],
        output_format: OutputFormat = OutputFormat.DASHBOARD
    ) -> str:
        """
        Format signals to specified output format.

        Returns formatted string.
        """
        if output_format == OutputFormat.DASHBOARD:
            return self._format_dashboard(signals)
        elif output_format == OutputFormat.TELEGRAM:
            return self._format_telegram(signals)
        elif output_format == OutputFormat.CSV:
            return self._format_csv(signals)
        elif output_format == OutputFormat.JSON:
            return self._format_json(signals)
        elif output_format == OutputFormat.PLAIN_TEXT:
            return self._format_plain_text(signals)
        elif output_format == OutputFormat.SLACK:
            return self._format_slack(signals)
        else:
            raise ValueError(f"Unknown format: {output_format}")

    def _format_dashboard(self, signals: List[RankedSignal]) -> str:
        """Format for dashboard display."""
        if not signals:
            return "No signals to display"

        lines = []
        lines.append("=" * 80)
        lines.append(f"  SIGNAL DASHBOARD - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(f"  {len(signals)} active signals")
        lines.append("=" * 80)
        lines.append("")

        for ranked in signals[:self.config.dashboard_max_signals]:
            signal = ranked.signal
            div = signal.divergence

            # Tier indicator
            tier_display = self._get_tier_display(ranked.tier)

            lines.append(f"{tier_display} #{ranked.rank} | Score: {ranked.effective_score:.1f}")
            lines.append("-" * 60)

            # Market info
            markets = list(div.market_ids)
            lines.append(f"  Markets: {' <-> '.join(markets)}")
            lines.append(f"  Type: {div.divergence_type.value}")
            lines.append(f"  Divergence: {div.divergence_pct:.2f}%")

            if div.is_arbitrage:
                lines.append(f"  ⚡ ARBITRAGE OPPORTUNITY")

            if self.config.dashboard_show_details:
                # Component scores
                lines.append("  Component Scores:")
                for name, comp in signal.component_scores.items():
                    lines.append(f"    {name}: {comp.score:.1f} (weight: {comp.weight:.1f})")

                # Recommendation
                if hasattr(signal, 'recommendation') and signal.recommendation:
                    rec = signal.recommendation
                    lines.append(f"  Recommendation: {rec.action.value}")
                    if hasattr(rec, 'urgency'):
                        lines.append(f"  Urgency: {rec.urgency.value}")

            lines.append("")

        if len(signals) > self.config.dashboard_max_signals:
            lines.append(f"... and {len(signals) - self.config.dashboard_max_signals} more signals")

        return "\n".join(lines)

    def _format_telegram(self, signals: List[RankedSignal]) -> str:
        """Format for Telegram message."""
        if not signals:
            return "📊 No active signals"

        parts = []

        # Header
        if self.config.telegram_emoji_enabled:
            parts.append(f"🔔 *{len(signals)} Trading Signals*")
        else:
            parts.append(f"** {len(signals)} Trading Signals **")
        parts.append(f"_{datetime.utcnow().strftime('%H:%M:%S')} UTC_")
        parts.append("")

        for ranked in signals:
            signal = ranked.signal
            div = signal.divergence

            # Emoji based on tier
            tier_emoji = self._get_tier_emoji(ranked.tier)

            # Signal header
            parts.append(f"{tier_emoji} *#{ranked.rank}* Score: {ranked.effective_score:.0f}")

            # Markets
            markets = list(div.market_ids)
            if len(markets) == 2:
                parts.append(f"📈 {markets[0]}")
                parts.append(f"📉 {markets[1]}")
            else:
                parts.append(f"📊 {', '.join(markets[:3])}")

            # Key metrics
            parts.append(f"Δ {div.divergence_pct:.1f}%")

            if div.is_arbitrage:
                parts.append("⚡ *ARBITRAGE*")

            # Recommendation if available
            if hasattr(signal, 'recommendation') and signal.recommendation:
                rec = signal.recommendation
                action_emoji = self._get_action_emoji(rec.action)
                parts.append(f"{action_emoji} {rec.action.value}")

            parts.append("")

            # Check message length
            current_text = "\n".join(parts)
            if len(current_text) > self.config.telegram_max_length - 200:
                parts.append(f"_...{len(signals) - signals.index(ranked)} more_")
                break

        return "\n".join(parts)

    def _format_csv(self, signals: List[RankedSignal]) -> str:
        """Format as CSV."""
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.config.csv_delimiter)

        if self.config.csv_include_header:
            writer.writerow([
                "rank",
                "tier",
                "effective_score",
                "overall_score",
                "market_a",
                "market_b",
                "divergence_type",
                "divergence_pct",
                "is_arbitrage",
                "timestamp",
                "divergence_id",
            ])

        for ranked in signals:
            signal = ranked.signal
            div = signal.divergence
            markets = list(div.market_ids)

            writer.writerow([
                ranked.rank,
                ranked.tier.value,
                round(ranked.effective_score, self.config.decimal_places),
                round(signal.overall_score, self.config.decimal_places),
                markets[0] if len(markets) > 0 else "",
                markets[1] if len(markets) > 1 else "",
                div.divergence_type.value,
                round(div.divergence_pct, self.config.decimal_places),
                div.is_arbitrage,
                signal.scored_at.isoformat() if self.config.include_timestamps else "",
                signal.divergence.id,
            ])

        return output.getvalue()

    def _format_json(self, signals: List[RankedSignal]) -> str:
        """Format as JSON."""
        data = {
            "signals": [],
        }

        if self.config.json_include_metadata:
            data["metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "count": len(signals),
            }

        for ranked in signals:
            signal = ranked.signal
            div = signal.divergence

            signal_data = {
                "rank": ranked.rank,
                "tier": ranked.tier.value,
                "effective_score": round(ranked.effective_score, self.config.decimal_places),
                "overall_score": round(signal.overall_score, self.config.decimal_places),
                "divergence_id": signal.divergence.id,
                "divergence": {
                    "type": div.divergence_type.value,
                    "percentage": round(div.divergence_pct, self.config.decimal_places),
                    "market_ids": list(div.market_ids),
                    "is_arbitrage": div.is_arbitrage,
                },
                "component_scores": {
                    name: {
                        "score": round(comp.score, self.config.decimal_places),
                        "weight": comp.weight,
                    }
                    for name, comp in signal.component_scores.items()
                },
            }

            if self.config.include_timestamps:
                signal_data["timestamp"] = signal.scored_at.isoformat()

            if hasattr(signal, 'recommendation') and signal.recommendation:
                rec = signal.recommendation
                signal_data["recommendation"] = {
                    "action": rec.action.value,
                }
                if hasattr(rec, 'urgency'):
                    signal_data["recommendation"]["urgency"] = rec.urgency.value

            if ranked.rank_factors:
                signal_data["rank_factors"] = {
                    k: round(v, self.config.decimal_places) if isinstance(v, float) else v
                    for k, v in ranked.rank_factors.items()
                }

            data["signals"].append(signal_data)

        if self.config.json_pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)

    def _format_plain_text(self, signals: List[RankedSignal]) -> str:
        """Format as plain text (for logs)."""
        if not signals:
            return "No signals"

        lines = []
        lines.append(f"Signals ({len(signals)}) at {datetime.utcnow().isoformat()}")
        lines.append("-" * 40)

        for ranked in signals:
            signal = ranked.signal
            div = signal.divergence
            markets = list(div.market_ids)

            lines.append(
                f"#{ranked.rank} [{ranked.tier.value}] "
                f"Score:{ranked.effective_score:.1f} "
                f"Div:{div.divergence_pct:.1f}% "
                f"Markets:{'/'.join(markets[:2])} "
                f"{'[ARB]' if div.is_arbitrage else ''}"
            )

        return "\n".join(lines)

    def _format_slack(self, signals: List[RankedSignal]) -> str:
        """
        Format for Slack webhook (JSON blocks format).

        Returns JSON string for Slack API.
        """
        blocks = []

        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🔔 {len(signals)} Trading Signals",
            }
        })

        blocks.append({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f"Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            }]
        })

        blocks.append({"type": "divider"})

        for ranked in signals[:10]:  # Slack has block limits
            signal = ranked.signal
            div = signal.divergence
            markets = list(div.market_ids)

            tier_emoji = self._get_tier_emoji(ranked.tier)

            # Signal block
            fields = [
                {
                    "type": "mrkdwn",
                    "text": f"*Score:* {ranked.effective_score:.0f}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Tier:* {tier_emoji} {ranked.tier.value}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Divergence:* {div.divergence_pct:.1f}%"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Type:* {div.divergence_type.value}"
                },
            ]

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*#{ranked.rank}* {' ↔ '.join(markets[:2])}"
                           + (" ⚡ *ARBITRAGE*" if div.is_arbitrage else "")
                },
                "fields": fields
            })

            blocks.append({"type": "divider"})

        return json.dumps({"blocks": blocks})

    def _get_tier_display(self, tier: SignalTier) -> str:
        """Get display string for tier."""
        displays = {
            SignalTier.CRITICAL: "[CRITICAL]",
            SignalTier.HIGH: "[HIGH]    ",
            SignalTier.MEDIUM: "[MEDIUM]  ",
            SignalTier.LOW: "[LOW]     ",
            SignalTier.IGNORE: "[IGNORE]  ",
        }
        return displays.get(tier, "[UNKNOWN] ")

    def _get_tier_emoji(self, tier: SignalTier) -> str:
        """Get emoji for tier."""
        if not self.config.telegram_emoji_enabled:
            return ""
        emojis = {
            SignalTier.CRITICAL: "🔴",
            SignalTier.HIGH: "🟠",
            SignalTier.MEDIUM: "🟡",
            SignalTier.LOW: "🟢",
            SignalTier.IGNORE: "⚪",
        }
        return emojis.get(tier, "⚪")

    def _get_action_emoji(self, action: RecommendedAction) -> str:
        """Get emoji for action."""
        if not self.config.telegram_emoji_enabled:
            return ""
        emojis = {
            RecommendedAction.STRONG_BUY: "🚀",
            RecommendedAction.BUY: "✅",
            RecommendedAction.WATCH: "👀",
            RecommendedAction.PASS: "⏭️",
        }
        return emojis.get(action, "")

    def format_single(
        self,
        ranked: RankedSignal,
        output_format: OutputFormat = OutputFormat.TELEGRAM
    ) -> str:
        """Format a single signal."""
        return self.format([ranked], output_format)

    def format_summary(
        self,
        signals: List[RankedSignal],
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT
    ) -> str:
        """Format a summary of signals (counts by tier)."""
        by_tier: Dict[SignalTier, int] = {}
        for tier in SignalTier:
            count = sum(1 for s in signals if s.tier == tier)
            if count > 0:
                by_tier[tier] = count

        if output_format == OutputFormat.TELEGRAM:
            parts = ["📊 *Signal Summary*", ""]
            for tier, count in sorted(by_tier.items(), key=lambda x: list(SignalTier).index(x[0])):
                emoji = self._get_tier_emoji(tier)
                parts.append(f"{emoji} {tier.value}: {count}")
            return "\n".join(parts)

        elif output_format == OutputFormat.JSON:
            return json.dumps({
                "summary": {tier.value: count for tier, count in by_tier.items()},
                "total": len(signals),
                "timestamp": datetime.utcnow().isoformat(),
            })

        else:
            lines = [f"Signal Summary ({len(signals)} total):"]
            for tier, count in sorted(by_tier.items(), key=lambda x: list(SignalTier).index(x[0])):
                lines.append(f"  {tier.value}: {count}")
            return "\n".join(lines)


class AlertFormatter:
    """
    Specialized formatter for urgent alerts.

    Produces concise, attention-grabbing messages for critical signals.
    """

    def __init__(self, emoji_enabled: bool = True):
        self.emoji_enabled = emoji_enabled

    def format_alert(self, ranked: RankedSignal) -> str:
        """Format an urgent alert."""
        signal = ranked.signal
        div = signal.divergence
        markets = list(div.market_ids)

        if self.emoji_enabled:
            alert = "🚨 "
        else:
            alert = "ALERT: "

        alert += f"SIGNAL #{ranked.rank}\n"
        alert += f"Score: {ranked.effective_score:.0f} | {ranked.tier.value.upper()}\n"
        alert += f"Markets: {' ↔ '.join(markets[:2])}\n"
        alert += f"Divergence: {div.divergence_pct:.1f}%\n"

        if div.is_arbitrage:
            alert += "⚡ ARBITRAGE DETECTED\n" if self.emoji_enabled else "** ARBITRAGE **\n"

        if hasattr(signal, 'recommendation') and signal.recommendation:
            alert += f"Action: {signal.recommendation.action.value}\n"

        return alert

    def format_batch_alert(
        self,
        signals: List[RankedSignal],
        threshold_tier: SignalTier = SignalTier.HIGH
    ) -> Optional[str]:
        """
        Format batch alert for signals above threshold tier.

        Returns None if no signals meet threshold.
        """
        tier_order = [
            SignalTier.CRITICAL,
            SignalTier.HIGH,
            SignalTier.MEDIUM,
            SignalTier.LOW,
            SignalTier.IGNORE,
        ]
        threshold_idx = tier_order.index(threshold_tier)
        alertable = [s for s in signals if tier_order.index(s.tier) <= threshold_idx]

        if not alertable:
            return None

        if self.emoji_enabled:
            alert = f"🚨 {len(alertable)} URGENT SIGNALS\n\n"
        else:
            alert = f"ALERT: {len(alertable)} URGENT SIGNALS\n\n"

        for ranked in alertable[:5]:
            signal = ranked.signal
            div = signal.divergence
            markets = list(div.market_ids)

            tier_emoji = "🔴" if ranked.tier == SignalTier.CRITICAL else "🟠"
            if not self.emoji_enabled:
                tier_emoji = f"[{ranked.tier.value}]"

            alert += f"{tier_emoji} #{ranked.rank} {div.divergence_pct:.1f}% "
            alert += f"{'/'.join(markets[:2])}\n"

        if len(alertable) > 5:
            alert += f"\n...and {len(alertable) - 5} more"

        return alert

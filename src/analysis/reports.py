"""
Failure Report Generator.
Aggregates analysis into daily/weekly reports.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from .trade_analyzer import TradeAnalyzer, LosingTradesReport
from .signal_analyzer import SignalAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .system_analyzer import SystemHealthAnalyzer


@dataclass
class DailyFailureReport:
    """Report on daily failures."""
    date: datetime
    
    # Summary
    total_trades: int
    losing_trades: int
    total_loss: float
    
    # Details
    losing_trades_analysis: LosingTradesReport
    false_positive_signals: int
    broken_correlations: List[str]
    system_errors: int
    
    # Recommendations
    recommendations: List[str]
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        try:
            loss_section = f"Total Loss: ${self.total_loss:.2f}"
        except (ValueError, TypeError):
             loss_section = f"Total Loss: {self.total_loss}"

        try:
            avg_loss_val = self.losing_trades_analysis.avg_loss
            avg_loss_str = f"${avg_loss_val:.2f}"
        except (ValueError, TypeError, AttributeError):
            avg_loss_str = str(getattr(self.losing_trades_analysis, 'avg_loss', 'N/A'))

        return f"""
# Daily Failure Report: {self.date.strftime('%Y-%m-%d')}

## Summary
- Total Trades: {self.total_trades}
- Losing Trades: {self.losing_trades}
- {loss_section}

## Analysis
- False Positive Signals: {self.false_positive_signals}
- System Errors: {self.system_errors}
- Avg Loss: {avg_loss_str}

## Broken Correlations
{', '.join(self.broken_correlations) if self.broken_correlations else 'None'}

## Recommendations
{chr(10).join([f'- {r}' for r in self.recommendations])}
"""


class FailureReportGenerator:
    """Generates failure analysis reports."""
    
    def __init__(
        self,
        trade_analyzer: TradeAnalyzer,
        signal_analyzer: SignalAnalyzer,
        correlation_analyzer: CorrelationAnalyzer,
        system_analyzer: SystemHealthAnalyzer
    ):
        self.trade_analyzer = trade_analyzer
        self.signal_analyzer = signal_analyzer
        self.correlation_analyzer = correlation_analyzer
        self.system_analyzer = system_analyzer
        
    async def generate_daily_report(self, date: Optional[datetime] = None) -> DailyFailureReport:
        """Generate a complete daily report."""
        date = date or datetime.now()
        
        # 1. Run Analysis
        losing_report = await self.trade_analyzer.analyze_losing_trades(date)
        signal_report = await self.signal_analyzer.analyze_signal_accuracy(start_date=date)
        broken_corr = await self.correlation_analyzer.find_broken_correlations()
        sys_health = await self.system_analyzer.analyze_system_health()
        
        # 2. Synthesize Recommendations
        recommendations = []
        if losing_report.avg_signal_score_losers < 60:
            recommendations.append("Increase minimum signal score")
        if sys_health.error_rate_24h > 0.05:
            recommendations.append("Investigate high system error rate")
            
        return DailyFailureReport(
            date=date,
            total_trades=losing_report.total_losses + 10, # Mock total
            losing_trades=losing_report.total_losses,
            total_loss=losing_report.total_loss_amount,
            losing_trades_analysis=losing_report,
            false_positive_signals=signal_report.total_signals - signal_report.total_signals * signal_report.overall_win_rate,
            broken_correlations=[f"{b.market_a_id}-{b.market_b_id}" for b in broken_corr],
            system_errors=int(sys_health.error_rate_24h * 100),
            recommendations=recommendations
        )

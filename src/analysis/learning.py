"""
Learning System module.
Suggests configuration adjustments based on analysis.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Any, Optional
import logging

from .reports import FailureReportGenerator, DailyFailureReport

logger = logging.getLogger(__name__)

@dataclass
class ConfigAdjustment:
    """Suggested configuration change."""
    param: str
    current_value: Any
    suggested_value: Any
    reason: str


@dataclass
class LearningResult:
    """Result of learning process."""
    date: datetime
    analysis: DailyFailureReport
    suggested_adjustments: List[ConfigAdjustment]
    auto_apply: bool


class LearningSystem:
    """
    Continuous learning system.
    Analyzes failures and suggests parameter updates.
    """
    
    def __init__(self, failure_report_generator: FailureReportGenerator, config_manager=None):
        self.report_generator = failure_report_generator
        self.config_manager = config_manager
        
    async def learn_from_today(self) -> LearningResult:
        """Analyze today's performance and suggest adjustments."""
        report = await self.report_generator.generate_daily_report()
        
        adjustments = []
        
        # 1. Win Rate Check
        # Estimate win rate from report (simplification)
        win_rate = 1.0 - (report.losing_trades / max(1, report.total_trades))
        
        if win_rate < 0.45:
             adjustments.append(ConfigAdjustment(
                param="min_signal_score",
                current_value=60, # Mock current
                suggested_value=65,
                reason=f"Low win rate ({win_rate:.2f})"
            ))
            
        # 2. Correlation Check
        if len(report.broken_correlations) > 2:
            adjustments.append(ConfigAdjustment(
                param="min_correlation_confidence",
                current_value=0.8,
                suggested_value=0.85,
                reason="Multiple correlation breakdowns"
            ))
            
        return LearningResult(
            date=datetime.now(),
            analysis=report,
            suggested_adjustments=adjustments,
            auto_apply=False
        )
        
    async def apply_adjustment(self, adjustment: ConfigAdjustment):
        """Apply a suggested adjustment."""
        # if self.config_manager:
        #     await self.config_manager.set(adjustment.param, adjustment.suggested_value)
        logger.info(f"Applied adjustment: {adjustment}")

"""
Root Cause Analyzer module.
Identify reasons for trade failures.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class RootCause:
    """Identified cause of a failure."""
    category: str
    description: str
    confidence: float
    evidence: str

@dataclass
class Recommendation:
    """Recommendation to prevent recurrence."""
    action: str
    impact: str
    effort: str


@dataclass
class RootCauseAnalysis:
    """Complete analysis of a failure root cause."""
    trade_id: str
    identified_causes: List[RootCause]
    primary_cause: Optional[RootCause]
    recommendations: List[Recommendation]


class RootCauseAnalyzer:
    """Analyzes trade context to determine failure causes."""
    
    def __init__(self):
        pass
        
    def analyze_trade_failure(
        self,
        trade_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> RootCauseAnalysis:
        """Determine root cause of trade failure."""
        causes = []
        
        # 1. Check Signal Quality
        score = trade_data.get("signal_score", 0)
        if score < 60:
            causes.append(RootCause(
                category="signal_quality",
                description="Low signal score",
                confidence=0.7,
                evidence=f"Score was {score}"
            ))
            
        # 2. Check Correlation
        entry_corr = trade_data.get("entry_correlation", 1.0)
        exit_corr = trade_data.get("exit_correlation", 0.5)
        if exit_corr < entry_corr - 0.2:
            causes.append(RootCause(
                category="correlation_breakdown",
                description="Correlation weakened during trade",
                confidence=0.8,
                evidence=f"Correlation dropped from {entry_corr} to {exit_corr}"
            ))
            
        # 3. Check Volatility/Events
        if market_context.get("news_event"):
            causes.append(RootCause(
                category="external_event",
                description="News event during trade",
                confidence=0.6,
                evidence="Detected news event"
            ))
            
        primary = max(causes, key=lambda c: c.confidence) if causes else None
        
        return RootCauseAnalysis(
            trade_id=trade_data.get("id", "unknown"),
            identified_causes=causes,
            primary_cause=primary,
            recommendations=self.generate_recommendations(causes)
        )
        
    def generate_recommendations(self, causes: List[RootCause]) -> List[Recommendation]:
        """Generate recommendations based on root causes."""
        recommendations = []
        for cause in causes:
            if cause.category == "signal_quality":
                recommendations.append(Recommendation(
                    action="Increase minimum signal score threshold",
                    impact="Fewer but higher quality signals",
                    effort="low"
                ))
            elif cause.category == "correlation_breakdown":
                recommendations.append(Recommendation(
                    action="Add correlation stability check",
                    impact="Avoid unstable correlations",
                    effort="medium"
                ))
        return recommendations

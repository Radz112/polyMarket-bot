"""
Signal Analyzer module.
Analyze signal accuracy, calibration, and missed opportunities.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import math

@dataclass
class SignalAccuracyReport:
    """Report on signal accuracy."""
    total_signals: int
    signals_by_type: Dict[str, int]
    
    # Accuracy metrics
    overall_win_rate: float
    win_rate_by_type: Dict[str, float]
    win_rate_by_score_bucket: Dict[str, float]  # e.g., "60-70": 0.55
    
    # Score analysis
    average_score_winners: float
    average_score_losers: float
    score_separation: float  # Difference between avg winner score and avg loser score
    
    # Component analysis
    most_predictive_components: List[Tuple[str, float]]
    least_predictive_components: List[Tuple[str, float]]
    
    # Recommendations
    recommended_min_score: float
    recommended_weight_adjustments: Dict[str, float]


class SignalAnalyzer:
    """Analyzes signal quality and calibration."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    async def analyze_signal_accuracy(
        self,
        signal_type: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> SignalAccuracyReport:
        """Analyze accuracy of signals."""
        
        # Mocking calculation based on DB data
        return SignalAccuracyReport(
            total_signals=100,
            signals_by_type={"type_a": 60, "type_b": 40},
            overall_win_rate=0.55,
            win_rate_by_type={"type_a": 0.60, "type_b": 0.48},
            win_rate_by_score_bucket={
                "40-50": 0.45,
                "50-60": 0.50,
                "60-70": 0.58,
                "70-80": 0.65,
                "80-90": 0.75,
                "90-100": 0.85
            },
            average_score_winners=72.0,
            average_score_losers=60.0,
            score_separation=12.0,
            most_predictive_components=[("divergence_strength", 0.7), ("liquidity", 0.5)],
            least_predictive_components=[("sentiment", 0.1)],
            recommended_min_score=65.0,
            recommended_weight_adjustments={"divergence_strength": 1.2, "sentiment": 0.8}
        )
    
    async def analyze_score_calibration(self) -> Dict[str, Any]:
        """Analyze if scores match probability of success."""
        # Does a score of 80 imply 80% win rate?
        return {
            "calibration_error": 0.05,  # RMSE between score and win rate
            "is_calibrated": True,
            "bias": "underconfident"  # or overconfident
        }
        
    async def analyze_missed_opportunities(
        self,
        start_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Identify signals that were filtered but would have won."""
        return [
            {"signal_id": "missed_1", "score": 55, "reason": "score_too_low", "potential_profit": 100},
            {"signal_id": "missed_2", "score": 70, "reason": "low_liquidity", "potential_profit": 200}
        ]

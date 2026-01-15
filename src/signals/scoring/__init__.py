"""
Signal scoring system.

Scores and ranks divergence signals by quality and tradability.
Provides trading recommendations based on configurable criteria.
"""
from src.signals.scoring.types import (
    ScoredSignal,
    ComponentScore,
    ScoringConfig,
    RecommendedAction,
    Urgency,
    BacktestResult,
    ScoreDistribution,
)
from src.signals.scoring.scorer import SignalScorer
from src.signals.scoring.calibrator import ScoreCalibrator, SignalOutcome, CalibrationResult
from src.signals.scoring.recommender import (
    ActionRecommender,
    TradingRecommendation,
    PortfolioConstraints,
)

__all__ = [
    # Types
    "ScoredSignal",
    "ComponentScore",
    "ScoringConfig",
    "RecommendedAction",
    "Urgency",
    "BacktestResult",
    "ScoreDistribution",
    # Scorer
    "SignalScorer",
    # Calibrator
    "ScoreCalibrator",
    "SignalOutcome",
    "CalibrationResult",
    # Recommender
    "ActionRecommender",
    "TradingRecommendation",
    "PortfolioConstraints",
]

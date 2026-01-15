"""
Signal scoring types and data structures.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List, Any

from src.signals.divergence.types import Divergence


class RecommendedAction(str, Enum):
    """Trading action recommendations."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    PASS = "PASS"


class Urgency(str, Enum):
    """Signal urgency levels."""
    IMMEDIATE = "IMMEDIATE"  # Trade now or miss it
    SOON = "SOON"           # Trade within minutes
    WATCH = "WATCH"         # Monitor for better entry


@dataclass
class ComponentScore:
    """Individual scoring component result."""
    name: str
    score: float           # 0-100
    weight: float          # 0-1
    weighted_score: float  # score * weight
    explanation: str       # Why this score?
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredSignal:
    """
    A divergence signal with computed scores and trading recommendations.

    Combines the raw divergence data with quality assessment scores
    and actionable trading recommendations.
    """
    # Original divergence
    divergence: Divergence

    # Timestamp
    scored_at: datetime = field(default_factory=datetime.utcnow)

    # Scores
    overall_score: float = 0.0  # 0-100
    component_scores: Dict[str, ComponentScore] = field(default_factory=dict)

    # Ranking metadata (set by ranker)
    rank: Optional[int] = None
    percentile: Optional[float] = None

    # Trading recommendation
    recommended_action: RecommendedAction = RecommendedAction.PASS
    recommended_size: float = 0.0
    recommended_price: float = 0.0

    # Risk metrics
    expected_profit: float = 0.0
    expected_loss: float = 0.0
    probability_of_profit: float = 0.5
    sharpe_estimate: float = 0.0

    # Timing
    urgency: Urgency = Urgency.WATCH
    estimated_window_seconds: int = 300  # Default 5 minutes

    # Explainability
    score_explanation: List[str] = field(default_factory=list)

    def __lt__(self, other: "ScoredSignal") -> bool:
        """For sorting by score (ascending)."""
        return self.overall_score < other.overall_score

    def __gt__(self, other: "ScoredSignal") -> bool:
        """For sorting by score (descending)."""
        return self.overall_score > other.overall_score

    def get_component_breakdown(self) -> Dict[str, float]:
        """Get simple dict of component name -> score."""
        return {name: cs.score for name, cs in self.component_scores.items()}

    def get_weighted_breakdown(self) -> Dict[str, float]:
        """Get dict of component name -> weighted contribution."""
        return {name: cs.weighted_score for name, cs in self.component_scores.items()}

    def explain(self) -> str:
        """Generate human-readable explanation of the score."""
        lines = [
            f"Signal Score: {self.overall_score:.1f}/100 ({self.recommended_action.value})",
            f"Urgency: {self.urgency.value}",
            "",
            "Component Breakdown:",
        ]

        # Sort by weighted contribution
        sorted_components = sorted(
            self.component_scores.values(),
            key=lambda x: x.weighted_score,
            reverse=True
        )

        for cs in sorted_components:
            lines.append(
                f"  {cs.name}: {cs.score:.1f} (weight: {cs.weight:.0%}, "
                f"contribution: {cs.weighted_score:.1f})"
            )
            lines.append(f"    -> {cs.explanation}")

        if self.score_explanation:
            lines.append("")
            lines.append("Key Factors:")
            for exp in self.score_explanation:
                lines.append(f"  - {exp}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "divergence_id": self.divergence.id,
            "divergence_type": self.divergence.divergence_type.value,
            "market_ids": self.divergence.market_ids,
            "scored_at": self.scored_at.isoformat(),
            "overall_score": self.overall_score,
            "component_scores": {
                name: {
                    "score": cs.score,
                    "weight": cs.weight,
                    "weighted_score": cs.weighted_score,
                    "explanation": cs.explanation,
                }
                for name, cs in self.component_scores.items()
            },
            "rank": self.rank,
            "percentile": self.percentile,
            "recommended_action": self.recommended_action.value,
            "recommended_size": self.recommended_size,
            "recommended_price": self.recommended_price,
            "expected_profit": self.expected_profit,
            "expected_loss": self.expected_loss,
            "probability_of_profit": self.probability_of_profit,
            "sharpe_estimate": self.sharpe_estimate,
            "urgency": self.urgency.value,
            "estimated_window_seconds": self.estimated_window_seconds,
            "score_explanation": self.score_explanation,
        }


@dataclass
class ScoringConfig:
    """Configuration for signal scoring."""

    # Component weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "divergence_size": 0.25,
        "liquidity": 0.20,
        "confidence": 0.20,
        "time_sensitivity": 0.15,
        "historical_accuracy": 0.10,
        "risk_reward": 0.10,
    })

    # Action thresholds
    action_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_BUY": 80.0,
        "BUY": 60.0,
        "WATCH": 40.0,
        "PASS": 0.0,
    })

    # Divergence size scoring thresholds (in cents/dollars)
    divergence_thresholds: List[tuple] = field(default_factory=lambda: [
        # (threshold, min_score, max_score)
        (0.01, 0, 20),      # < 1¢
        (0.02, 20, 40),     # 1-2¢
        (0.04, 40, 60),     # 2-4¢
        (0.06, 60, 80),     # 4-6¢
        (float('inf'), 80, 100),  # > 6¢
    ])

    # Liquidity scoring thresholds (in dollars)
    liquidity_thresholds: List[tuple] = field(default_factory=lambda: [
        # (threshold, min_score, max_score)
        (50, 0, 20),        # < $50
        (200, 20, 50),      # $50-200
        (500, 50, 70),      # $200-500
        (1000, 70, 85),     # $500-1000
        (float('inf'), 85, 100),  # > $1000
    ])

    # Minimum score for arbitrage signals
    arbitrage_min_score: float = 50.0

    # Historical lookback for accuracy scoring
    historical_lookback_days: int = 30

    # Risk-free rate for Sharpe calculation (annual)
    risk_free_rate: float = 0.05

    # Maximum position size as fraction of bankroll
    max_position_fraction: float = 0.05

    # Kelly fraction (fraction of Kelly criterion to use)
    kelly_fraction: float = 0.25

    def validate(self) -> bool:
        """Validate configuration."""
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        return True


@dataclass
class BacktestResult:
    """Results from backtesting score thresholds."""
    min_score_threshold: float
    period_days: int

    # Counts
    total_signals: int
    signals_above_threshold: int

    # Performance
    win_count: int
    loss_count: int
    win_rate: float

    # Profit
    total_profit: float
    average_profit: float
    max_profit: float
    max_loss: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float

    # Timing
    average_hold_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_score_threshold": self.min_score_threshold,
            "period_days": self.period_days,
            "total_signals": self.total_signals,
            "signals_above_threshold": self.signals_above_threshold,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "total_profit": self.total_profit,
            "average_profit": self.average_profit,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "average_hold_seconds": self.average_hold_seconds,
        }


@dataclass
class ScoreDistribution:
    """Statistics about score distribution."""
    count: int
    mean: float
    median: float
    std: float
    min_score: float
    max_score: float
    percentiles: Dict[int, float]  # {10: score, 25: score, 50: score, ...}

    # Distribution by action
    action_counts: Dict[str, int]

    # Score vs outcome correlation (if outcome data available)
    score_outcome_correlation: Optional[float] = None
    optimal_threshold: Optional[float] = None

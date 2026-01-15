"""
Score calibrator for tuning scoring weights based on outcomes.

Analyzes historical signal performance to optimize weights.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.signals.divergence.types import Divergence
from src.signals.scoring.types import (
    ScoredSignal,
    ScoringConfig,
    BacktestResult,
    ScoreDistribution,
)

logger = logging.getLogger(__name__)


@dataclass
class SignalOutcome:
    """Historical signal with known outcome."""
    divergence: Divergence
    signal_time: datetime
    score_at_detection: float
    component_scores: Dict[str, float]

    # Outcome data
    outcome: str  # "win", "loss", "expired"
    profit: float  # Positive or negative
    close_time: datetime
    convergence_seconds: int


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    improvement: float  # Improvement in objective metric
    sample_size: int
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]


class ScoreCalibrator:
    """
    Calibrates scoring weights based on historical outcomes.

    Uses historical signal data with known outcomes to:
    1. Analyze component-outcome correlations
    2. Find optimal weights
    3. Determine optimal action thresholds
    """

    def __init__(
        self,
        config: ScoringConfig = None,
        db_manager=None,
    ):
        self.config = config or ScoringConfig()
        self.db = db_manager

        # Historical outcomes for calibration
        self._outcomes: List[SignalOutcome] = []
        self._max_outcomes = 5000

    def add_outcome(self, outcome: SignalOutcome) -> None:
        """Add a signal outcome for calibration."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._max_outcomes:
            self._outcomes = self._outcomes[-self._max_outcomes:]

    async def load_outcomes_from_db(
        self,
        lookback_days: int = 30
    ) -> int:
        """Load historical outcomes from database."""
        if self.db is None:
            logger.warning("No database configured for calibration")
            return 0

        # TODO: Implement when signals_history table is available
        # This would query:
        # SELECT * FROM signals_history
        # WHERE signal_time > NOW() - INTERVAL ? DAY
        # AND outcome IS NOT NULL

        return 0

    def calibrate_weights(
        self,
        objective: str = "sharpe",
        min_samples: int = 100,
    ) -> Optional[CalibrationResult]:
        """
        Calibrate weights to optimize objective metric.

        Args:
            objective: What to optimize ("sharpe", "win_rate", "profit")
            min_samples: Minimum outcomes required for calibration

        Returns:
            CalibrationResult if successful, None if insufficient data
        """
        if len(self._outcomes) < min_samples:
            logger.warning(
                f"Insufficient data for calibration: {len(self._outcomes)} < {min_samples}"
            )
            return None

        # Calculate current metrics
        old_weights = self.config.weights.copy()
        metrics_before = self._calculate_metrics(old_weights)

        # Find optimal weights using gradient-free optimization
        new_weights = self._optimize_weights(objective)

        # Calculate metrics with new weights
        metrics_after = self._calculate_metrics(new_weights)

        # Calculate improvement
        if objective == "sharpe":
            improvement = metrics_after["sharpe_ratio"] - metrics_before["sharpe_ratio"]
        elif objective == "win_rate":
            improvement = metrics_after["win_rate"] - metrics_before["win_rate"]
        else:  # profit
            improvement = metrics_after["total_profit"] - metrics_before["total_profit"]

        return CalibrationResult(
            old_weights=old_weights,
            new_weights=new_weights,
            improvement=improvement,
            sample_size=len(self._outcomes),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

    def analyze_component_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze importance of each scoring component.

        Returns dict mapping component name to importance metrics.
        """
        if len(self._outcomes) < 20:
            return {}

        results = {}
        components = list(self.config.weights.keys())

        for component in components:
            # Get component scores and outcomes
            scores = [o.component_scores.get(component, 50) for o in self._outcomes]
            profits = [o.profit for o in self._outcomes]
            wins = [1 if o.outcome == "win" else 0 for o in self._outcomes]

            # Calculate correlation with profit
            profit_corr = self._pearson_correlation(scores, profits)

            # Calculate correlation with win/loss
            win_corr = self._pearson_correlation(scores, wins)

            # Calculate average score for winners vs losers
            winner_scores = [
                s for s, o in zip(scores, self._outcomes)
                if o.outcome == "win"
            ]
            loser_scores = [
                s for s, o in zip(scores, self._outcomes)
                if o.outcome == "loss"
            ]

            avg_winner = sum(winner_scores) / len(winner_scores) if winner_scores else 0
            avg_loser = sum(loser_scores) / len(loser_scores) if loser_scores else 0

            results[component] = {
                "profit_correlation": profit_corr,
                "win_correlation": win_corr,
                "avg_winner_score": avg_winner,
                "avg_loser_score": avg_loser,
                "discriminative_power": avg_winner - avg_loser,
            }

        return results

    def find_optimal_threshold(
        self,
        min_win_rate: float = 0.6,
    ) -> Tuple[float, BacktestResult]:
        """
        Find optimal score threshold for trade execution.

        Args:
            min_win_rate: Minimum acceptable win rate

        Returns:
            Tuple of (optimal_threshold, backtest_result)
        """
        # Test thresholds from 40 to 90
        best_threshold = 60.0
        best_result = None
        best_sharpe = float('-inf')

        for threshold in range(40, 91, 5):
            result = self.backtest_threshold(float(threshold))

            if result.win_rate >= min_win_rate and result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_threshold = float(threshold)
                best_result = result

        if best_result is None:
            # Fall back to threshold with best win rate
            best_result = self.backtest_threshold(60.0)
            best_threshold = 60.0

        return best_threshold, best_result

    def backtest_threshold(
        self,
        threshold: float,
        period_days: int = 30,
    ) -> BacktestResult:
        """
        Backtest a specific score threshold.

        Simulates trading only signals above the threshold.
        """
        # Filter to recent period
        cutoff = datetime.utcnow() - timedelta(days=period_days)
        recent = [o for o in self._outcomes if o.signal_time >= cutoff]

        if not recent:
            recent = self._outcomes  # Use all if none recent

        # Filter by threshold
        above_threshold = [o for o in recent if o.score_at_detection >= threshold]

        if not above_threshold:
            return self._empty_backtest_result(threshold, period_days)

        # Calculate metrics
        wins = [o for o in above_threshold if o.outcome == "win"]
        losses = [o for o in above_threshold if o.outcome == "loss"]

        win_count = len(wins)
        loss_count = len(losses)
        total_trades = win_count + loss_count

        win_rate = win_count / total_trades if total_trades > 0 else 0

        profits = [o.profit for o in above_threshold]
        total_profit = sum(profits)
        avg_profit = total_profit / len(above_threshold) if above_threshold else 0

        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0

        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe(profits)

        # Calculate max drawdown
        drawdown = self._calculate_max_drawdown(profits)

        # Average hold time
        hold_times = [o.convergence_seconds for o in above_threshold]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

        return BacktestResult(
            min_score_threshold=threshold,
            period_days=period_days,
            total_signals=len(recent),
            signals_above_threshold=len(above_threshold),
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            total_profit=total_profit,
            average_profit=avg_profit,
            max_profit=max_profit,
            max_loss=max_loss,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            average_hold_seconds=avg_hold,
        )

    def _optimize_weights(self, objective: str) -> Dict[str, float]:
        """
        Find optimal weights using coordinate descent.

        Simple optimization that adjusts each weight while holding others fixed.
        """
        components = list(self.config.weights.keys())
        current_weights = self.config.weights.copy()

        # Initial objective value
        best_score = self._evaluate_weights(current_weights, objective)

        # Coordinate descent iterations
        for iteration in range(10):
            improved = False

            for component in components:
                original = current_weights[component]

                # Try increasing
                current_weights[component] = min(0.5, original + 0.05)
                self._normalize_weights(current_weights)
                score = self._evaluate_weights(current_weights, objective)

                if score > best_score:
                    best_score = score
                    improved = True
                else:
                    # Try decreasing
                    current_weights[component] = max(0.05, original - 0.05)
                    self._normalize_weights(current_weights)
                    score = self._evaluate_weights(current_weights, objective)

                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        current_weights[component] = original
                        self._normalize_weights(current_weights)

            if not improved:
                break

        return current_weights

    def _evaluate_weights(
        self,
        weights: Dict[str, float],
        objective: str
    ) -> float:
        """Evaluate weights on the objective metric."""
        metrics = self._calculate_metrics(weights)

        if objective == "sharpe":
            return metrics["sharpe_ratio"]
        elif objective == "win_rate":
            return metrics["win_rate"]
        else:  # profit
            return metrics["total_profit"]

    def _calculate_metrics(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance metrics with given weights."""
        if not self._outcomes:
            return {
                "win_rate": 0.5,
                "total_profit": 0.0,
                "sharpe_ratio": 0.0,
            }

        # Recalculate scores with new weights
        scores = []
        for outcome in self._outcomes:
            score = sum(
                outcome.component_scores.get(comp, 50) * weight
                for comp, weight in weights.items()
            )
            scores.append(score)

        # Filter signals above 60 threshold
        threshold = 60.0
        above = [
            (s, o) for s, o in zip(scores, self._outcomes)
            if s >= threshold
        ]

        if not above:
            return {
                "win_rate": 0.5,
                "total_profit": 0.0,
                "sharpe_ratio": 0.0,
            }

        wins = sum(1 for _, o in above if o.outcome == "win")
        losses = sum(1 for _, o in above if o.outcome == "loss")
        total = wins + losses

        win_rate = wins / total if total > 0 else 0.5
        profits = [o.profit for _, o in above]
        total_profit = sum(profits)
        sharpe = self._calculate_sharpe(profits)

        return {
            "win_rate": win_rate,
            "total_profit": total_profit,
            "sharpe_ratio": sharpe,
        }

    def _normalize_weights(self, weights: Dict[str, float]) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] /= total

    def _calculate_sharpe(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio from profit series."""
        if len(profits) < 2:
            return 0.0

        mean_profit = sum(profits) / len(profits)

        variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
        std = math.sqrt(variance) if variance > 0 else 0.001

        # Annualized (assume ~1000 trades per year)
        return (mean_profit / std) * math.sqrt(1000) if std > 0 else 0

    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from profit series."""
        if not profits:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for profit in profits:
            cumulative += profit
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2 or n != len(y):
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)

    def _empty_backtest_result(
        self,
        threshold: float,
        period_days: int
    ) -> BacktestResult:
        """Create empty backtest result."""
        return BacktestResult(
            min_score_threshold=threshold,
            period_days=period_days,
            total_signals=0,
            signals_above_threshold=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            total_profit=0.0,
            average_profit=0.0,
            max_profit=0.0,
            max_loss=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            average_hold_seconds=0.0,
        )

    def get_weight_recommendations(self) -> Dict[str, Any]:
        """
        Get weight adjustment recommendations based on analysis.

        Returns human-readable recommendations.
        """
        importance = self.analyze_component_importance()

        if not importance:
            return {"status": "insufficient_data", "recommendations": []}

        recommendations = []

        # Find most and least predictive components
        sorted_by_power = sorted(
            importance.items(),
            key=lambda x: x[1]["discriminative_power"],
            reverse=True
        )

        # Recommend increasing weight for high-power components
        for comp, metrics in sorted_by_power[:2]:
            power = metrics["discriminative_power"]
            if power > 5:
                current = self.config.weights.get(comp, 0)
                recommendations.append({
                    "component": comp,
                    "action": "increase",
                    "reason": f"High discriminative power ({power:.1f})",
                    "current_weight": current,
                    "suggested_weight": min(0.35, current + 0.05),
                })

        # Recommend decreasing weight for low-power components
        for comp, metrics in sorted_by_power[-2:]:
            power = metrics["discriminative_power"]
            if power < 2:
                current = self.config.weights.get(comp, 0)
                recommendations.append({
                    "component": comp,
                    "action": "decrease",
                    "reason": f"Low discriminative power ({power:.1f})",
                    "current_weight": current,
                    "suggested_weight": max(0.05, current - 0.05),
                })

        return {
            "status": "success",
            "sample_size": len(self._outcomes),
            "component_importance": importance,
            "recommendations": recommendations,
        }

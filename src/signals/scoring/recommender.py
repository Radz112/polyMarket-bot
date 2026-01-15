"""
Action recommender for scored signals.

Converts scores into actionable trading recommendations with
position sizing, risk limits, and execution guidance.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import (
    ScoredSignal,
    ScoringConfig,
    RecommendedAction,
    Urgency,
)

logger = logging.getLogger(__name__)


@dataclass
class TradingRecommendation:
    """Detailed trading recommendation for a scored signal."""
    signal: ScoredSignal
    recommendation_time: datetime = field(default_factory=datetime.utcnow)

    # Action
    action: RecommendedAction = RecommendedAction.PASS
    action_description: str = ""

    # Position details
    trade_type: str = ""  # "BUY", "SELL", "SPREAD"
    markets: List[str] = field(default_factory=list)
    target_prices: Dict[str, float] = field(default_factory=dict)
    position_size: float = 0.0
    max_position_size: float = 0.0

    # Risk parameters
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    max_slippage_pct: float = 0.02  # 2% default

    # Timing
    urgency: Urgency = Urgency.WATCH
    execution_window_seconds: int = 300
    should_use_limit_order: bool = True

    # Expected outcomes
    expected_profit: float = 0.0
    expected_loss: float = 0.0
    probability_of_profit: float = 0.5
    risk_reward_ratio: float = 0.0

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def should_execute(self) -> bool:
        """Check if this recommendation should be executed."""
        return self.action in [RecommendedAction.STRONG_BUY, RecommendedAction.BUY]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "divergence_id": self.signal.divergence.id,
            "action": self.action.value,
            "action_description": self.action_description,
            "trade_type": self.trade_type,
            "markets": self.markets,
            "target_prices": self.target_prices,
            "position_size": self.position_size,
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_slippage_pct": self.max_slippage_pct,
            "urgency": self.urgency.value,
            "execution_window_seconds": self.execution_window_seconds,
            "should_use_limit_order": self.should_use_limit_order,
            "expected_profit": self.expected_profit,
            "expected_loss": self.expected_loss,
            "probability_of_profit": self.probability_of_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
            "warnings": self.warnings,
            "notes": self.notes,
        }


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio-level risk management."""
    max_total_position: float = 10000.0  # Maximum total position value
    max_single_position: float = 500.0   # Maximum single position
    max_position_pct: float = 0.05       # Max 5% of portfolio per trade
    max_daily_trades: int = 50           # Maximum trades per day
    max_concurrent_positions: int = 10   # Maximum simultaneous positions

    # Risk limits
    max_daily_loss: float = 500.0        # Stop trading after this loss
    max_drawdown_pct: float = 0.10       # Maximum 10% drawdown

    # Execution preferences
    prefer_limit_orders: bool = True
    min_time_between_trades: int = 5     # Seconds between trades


class ActionRecommender:
    """
    Converts scored signals into actionable trading recommendations.

    Applies portfolio constraints, risk limits, and execution preferences
    to generate detailed trade recommendations.
    """

    def __init__(
        self,
        config: ScoringConfig = None,
        constraints: PortfolioConstraints = None,
        bankroll: float = 10000.0,
    ):
        self.config = config or ScoringConfig()
        self.constraints = constraints or PortfolioConstraints()
        self.bankroll = bankroll

        # Track daily activity
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_time: Optional[datetime] = None
        self._active_positions: Dict[str, float] = {}

    def recommend(self, signal: ScoredSignal) -> TradingRecommendation:
        """
        Generate trading recommendation for a scored signal.

        Args:
            signal: Scored signal to generate recommendation for

        Returns:
            TradingRecommendation with action details
        """
        rec = TradingRecommendation(signal=signal)

        # Check portfolio-level constraints first
        constraint_warnings = self._check_constraints(signal)
        rec.warnings.extend(constraint_warnings)

        # If hard constraints violated, return PASS
        if self._has_hard_constraint_violation(constraint_warnings):
            rec.action = RecommendedAction.PASS
            rec.action_description = "Blocked by portfolio constraints"
            return rec

        # Set action based on signal score
        rec.action = signal.recommended_action
        rec.urgency = signal.urgency

        # Generate trade details based on divergence type
        self._populate_trade_details(rec, signal)

        # Calculate position sizing
        self._calculate_position_size(rec, signal)

        # Set risk parameters
        self._set_risk_parameters(rec, signal)

        # Set timing parameters
        self._set_timing_parameters(rec, signal)

        # Add execution notes
        self._add_execution_notes(rec, signal)

        # Build action description
        rec.action_description = self._build_action_description(rec)

        return rec

    def recommend_batch(
        self,
        signals: List[ScoredSignal],
        max_recommendations: int = 5,
    ) -> List[TradingRecommendation]:
        """
        Generate recommendations for multiple signals.

        Prioritizes signals by score and applies portfolio-level constraints
        across the batch to avoid over-concentration.

        Args:
            signals: List of scored signals
            max_recommendations: Maximum number of active recommendations

        Returns:
            List of TradingRecommendations, sorted by priority
        """
        # Sort by score descending
        sorted_signals = sorted(signals, key=lambda s: s.overall_score, reverse=True)

        recommendations = []
        total_position = 0.0
        markets_used = set()

        for signal in sorted_signals:
            if len(recommendations) >= max_recommendations:
                break

            rec = self.recommend(signal)

            # Skip if not actionable
            if not rec.should_execute():
                continue

            # Check for market overlap
            signal_markets = set(signal.divergence.market_ids)
            if signal_markets & markets_used:
                rec.warnings.append("Market overlap with higher-priority signal")
                rec.action = RecommendedAction.WATCH
                rec.action_description = "Waiting for higher-priority trade"

            # Check total position limit
            if total_position + rec.position_size > self.constraints.max_total_position:
                rec.position_size = max(
                    0,
                    self.constraints.max_total_position - total_position
                )
                if rec.position_size < 10:  # Minimum viable position
                    rec.action = RecommendedAction.PASS
                    rec.action_description = "Position limit reached"

            recommendations.append(rec)
            total_position += rec.position_size
            markets_used.update(signal_markets)

        return recommendations

    def _check_constraints(self, signal: ScoredSignal) -> List[str]:
        """Check portfolio constraints and return warnings."""
        warnings = []

        # Daily trade limit
        if self._daily_trades >= self.constraints.max_daily_trades:
            warnings.append("HARD: Daily trade limit reached")

        # Daily loss limit
        if self._daily_pnl <= -self.constraints.max_daily_loss:
            warnings.append("HARD: Daily loss limit reached")

        # Time between trades
        if self._last_trade_time:
            seconds_since = (datetime.utcnow() - self._last_trade_time).total_seconds()
            if seconds_since < self.constraints.min_time_between_trades:
                warnings.append("SOFT: Too soon since last trade")

        # Concurrent positions
        if len(self._active_positions) >= self.constraints.max_concurrent_positions:
            warnings.append("HARD: Maximum concurrent positions reached")

        # Check if already in this market
        for market_id in signal.divergence.market_ids:
            if market_id in self._active_positions:
                warnings.append(f"SOFT: Already have position in {market_id}")

        return warnings

    def _has_hard_constraint_violation(self, warnings: List[str]) -> bool:
        """Check if any hard constraints are violated."""
        return any(w.startswith("HARD:") for w in warnings)

    def _populate_trade_details(
        self,
        rec: TradingRecommendation,
        signal: ScoredSignal
    ) -> None:
        """Populate trade details based on divergence type."""
        divergence = signal.divergence

        rec.markets = divergence.market_ids.copy()
        rec.target_prices = divergence.current_prices.copy()

        dtype = divergence.divergence_type

        # Determine trade type
        if divergence.is_arbitrage:
            rec.trade_type = "ARBITRAGE"
            rec.notes.append("Risk-free arbitrage opportunity")
        elif dtype == DivergenceType.PRICE_SPREAD:
            rec.trade_type = "SPREAD"
            rec.notes.append("Price spread trade - buy cheap, sell expensive")
        elif dtype == DivergenceType.LEAD_LAG_OPPORTUNITY:
            rec.trade_type = "DIRECTIONAL"
            rec.notes.append("Lead-lag momentum trade")
        elif dtype == DivergenceType.CORRELATION_BREAK:
            rec.trade_type = "REVERSION"
            rec.notes.append("Betting on correlation reversion")
        else:
            direction = divergence.direction
            if "BUY" in direction:
                rec.trade_type = "BUY"
            elif "SELL" in direction:
                rec.trade_type = "SELL"
            else:
                rec.trade_type = "DIRECTIONAL"

    def _calculate_position_size(
        self,
        rec: TradingRecommendation,
        signal: ScoredSignal
    ) -> None:
        """Calculate recommended position size."""
        # Start with signal's recommended size
        base_size = signal.recommended_size

        # Apply constraints
        max_by_single = self.constraints.max_single_position
        max_by_pct = self.bankroll * self.constraints.max_position_pct
        max_by_liquidity = signal.divergence.max_executable_size

        # Score-based scaling
        score = signal.overall_score
        if score >= 80:
            score_mult = 1.0
        elif score >= 60:
            score_mult = 0.75
        elif score >= 40:
            score_mult = 0.5
        else:
            score_mult = 0.25

        # For arbitrage, allow larger positions
        if signal.divergence.is_arbitrage:
            score_mult = min(1.0, score_mult * 1.5)

        recommended = base_size * score_mult
        rec.position_size = min(
            recommended,
            max_by_single,
            max_by_pct,
            max_by_liquidity
        )
        rec.max_position_size = max_by_liquidity

    def _set_risk_parameters(
        self,
        rec: TradingRecommendation,
        signal: ScoredSignal
    ) -> None:
        """Set stop-loss and take-profit parameters."""
        divergence = signal.divergence
        profit_potential = divergence.profit_potential

        # For arbitrage, minimal stops needed
        if divergence.is_arbitrage:
            rec.stop_loss_pct = 0.01  # 1% just for safety
            rec.take_profit_pct = profit_potential
            rec.max_slippage_pct = profit_potential * 0.1  # 10% of profit
        else:
            # Dynamic stops based on divergence size
            rec.stop_loss_pct = max(0.05, profit_potential * 0.5)  # 5% min
            rec.take_profit_pct = max(0.03, profit_potential * 0.8)
            rec.max_slippage_pct = 0.02  # 2% max slippage

        # Set expected outcomes
        rec.expected_profit = signal.expected_profit
        rec.expected_loss = signal.expected_loss
        rec.probability_of_profit = signal.probability_of_profit

        # Calculate risk/reward ratio
        if rec.expected_loss > 0:
            rec.risk_reward_ratio = rec.expected_profit / rec.expected_loss
        else:
            rec.risk_reward_ratio = float('inf') if rec.expected_profit > 0 else 0

    def _set_timing_parameters(
        self,
        rec: TradingRecommendation,
        signal: ScoredSignal
    ) -> None:
        """Set execution timing parameters."""
        rec.execution_window_seconds = signal.estimated_window_seconds

        # Use limit orders for most cases
        rec.should_use_limit_order = self.constraints.prefer_limit_orders

        # For urgent arbitrage, may need market orders
        if signal.urgency == Urgency.IMMEDIATE and signal.divergence.is_arbitrage:
            rec.should_use_limit_order = False
            rec.notes.append("Consider market order for speed")

    def _add_execution_notes(
        self,
        rec: TradingRecommendation,
        signal: ScoredSignal
    ) -> None:
        """Add execution guidance notes."""
        divergence = signal.divergence
        dtype = divergence.divergence_type

        # Type-specific notes
        if dtype == DivergenceType.THRESHOLD_VIOLATION:
            rec.notes.append("Execute both legs simultaneously")
            rec.notes.append("Check for fee impact on profit")

        elif dtype == DivergenceType.INVERSE_SUM:
            rec.notes.append("Buy YES and NO to guarantee profit")
            rec.notes.append("Exit when sum normalizes to 1.0")

        elif dtype == DivergenceType.PRICE_SPREAD:
            if len(rec.markets) == 2:
                prices = list(divergence.current_prices.values())
                if len(prices) == 2:
                    cheap = min(prices)
                    expensive = max(prices)
                    rec.notes.append(f"Buy at {cheap:.2f}, sell at {expensive:.2f}")

        elif dtype == DivergenceType.LEAD_LAG_OPPORTUNITY:
            lag = divergence.metadata.get("lag_seconds", 60)
            rec.notes.append(f"Expected convergence in ~{lag}s")

        # Urgency-specific notes
        if signal.urgency == Urgency.IMMEDIATE:
            rec.warnings.append("Time-critical - act now or miss opportunity")
        elif signal.urgency == Urgency.SOON:
            rec.notes.append("Trade within next few minutes")

        # Liquidity notes
        liquidity_score = signal.component_scores.get("liquidity")
        if liquidity_score and liquidity_score.score < 50:
            rec.warnings.append("Low liquidity - expect slippage")
            rec.notes.append("Consider smaller position or limit orders")

    def _build_action_description(self, rec: TradingRecommendation) -> str:
        """Build human-readable action description."""
        if rec.action == RecommendedAction.PASS:
            return "Do not trade - signal below threshold"

        if rec.action == RecommendedAction.WATCH:
            return "Monitor for better entry"

        parts = []

        # Action type
        if rec.action == RecommendedAction.STRONG_BUY:
            parts.append("STRONG")

        parts.append(rec.trade_type)

        # Position
        if rec.position_size > 0:
            parts.append(f"${rec.position_size:.0f}")

        # Markets
        if rec.markets:
            parts.append(f"in {len(rec.markets)} market(s)")

        # Urgency
        if rec.urgency == Urgency.IMMEDIATE:
            parts.append("- URGENT")
        elif rec.urgency == Urgency.SOON:
            parts.append("- soon")

        return " ".join(parts)

    def record_trade(
        self,
        market_id: str,
        position_size: float,
        pnl: float = 0.0
    ) -> None:
        """
        Record a trade for constraint tracking.

        Call this when a trade is executed.
        """
        self._daily_trades += 1
        self._last_trade_time = datetime.utcnow()
        self._active_positions[market_id] = position_size
        self._daily_pnl += pnl

    def close_position(self, market_id: str, pnl: float) -> None:
        """Record closing a position."""
        if market_id in self._active_positions:
            del self._active_positions[market_id]
        self._daily_pnl += pnl

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of day)."""
        self._daily_trades = 0
        self._daily_pnl = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current recommender status."""
        return {
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl,
            "active_positions": len(self._active_positions),
            "trades_remaining": max(
                0,
                self.constraints.max_daily_trades - self._daily_trades
            ),
            "loss_headroom": max(
                0,
                self.constraints.max_daily_loss + self._daily_pnl
            ),
        }

"""
Risk/reward scoring component.

Scores signals based on risk-adjusted return potential.
"""
import math
from typing import Optional

from src.signals.divergence.types import Divergence, DivergenceType
from src.signals.scoring.types import ComponentScore, ScoringConfig


class RiskRewardScorer:
    """
    Scores divergences based on risk/reward profile.

    Considers:
    - Expected profit if correct
    - Expected loss if wrong
    - Probability of being correct
    - Kelly criterion calculations
    """

    # Win probability by type (base estimates)
    WIN_PROBABILITIES = {
        DivergenceType.THRESHOLD_VIOLATION: 0.98,   # Almost certain
        DivergenceType.INVERSE_SUM: 0.95,           # Very likely
        DivergenceType.PRICE_SPREAD: 0.75,          # Usually converges
        DivergenceType.LEAD_LAG_OPPORTUNITY: 0.65,  # Statistical edge
        DivergenceType.LAGGING_MARKET: 0.60,        # Often works
        DivergenceType.CORRELATION_BREAK: 0.50,     # Uncertain
    }

    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
        self.risk_free_rate = self.config.risk_free_rate
        self.kelly_fraction = self.config.kelly_fraction

    def score(self, divergence: Divergence) -> ComponentScore:
        """
        Score based on risk/reward profile.

        For arbitrage: 100 (no risk)
        For directional: based on expected value and Sharpe-like metrics
        """
        # For arbitrage, max score
        if divergence.is_arbitrage:
            return self._score_arbitrage(divergence)

        # For directional signals
        return self._score_directional(divergence)

    def _score_arbitrage(self, divergence: Divergence) -> ComponentScore:
        """Score an arbitrage opportunity (risk-free profit)."""
        profit = divergence.profit_potential

        # Scale score based on profit magnitude
        # Any positive profit is good, but larger is better
        if profit <= 0:
            base_score = 50  # Still arbitrage, but no profit?
        elif profit < 0.01:
            base_score = 80 + profit * 2000  # 80-100 for 0-1¢
        else:
            base_score = 100  # Max for 1¢+

        return ComponentScore(
            name="risk_reward",
            score=base_score,
            weight=self.config.weights.get("risk_reward", 0.10),
            weighted_score=0.0,
            explanation=f"Arbitrage: {profit*100:.1f}¢ risk-free profit",
            metadata={
                "is_arbitrage": True,
                "profit_potential": profit,
                "win_probability": 1.0,
                "expected_value": profit,
                "kelly_fraction": 1.0,  # Bet everything on sure things
                "sharpe_estimate": float('inf') if profit > 0 else 0,
            }
        )

    def _score_directional(self, divergence: Divergence) -> ComponentScore:
        """Score a directional (risky) signal."""
        dtype = divergence.divergence_type

        # Estimate win probability
        base_prob = self.WIN_PROBABILITIES.get(dtype, 0.5)
        win_prob = self._adjust_probability(divergence, base_prob)

        # Estimate potential profit and loss
        profit_if_win = divergence.profit_potential
        loss_if_lose = self._estimate_loss(divergence)

        # Calculate expected value
        expected_value = (win_prob * profit_if_win) - ((1 - win_prob) * loss_if_lose)

        # Calculate Kelly fraction
        kelly = self._calculate_kelly(win_prob, profit_if_win, loss_if_lose)

        # Calculate Sharpe-like ratio
        sharpe = self._estimate_sharpe(expected_value, profit_if_win, loss_if_lose, win_prob)

        # Combine into score
        # EV score: scaled 0-50 based on expected value
        ev_score = self._score_expected_value(expected_value)

        # Probability score: scaled 0-30 based on win probability
        prob_score = win_prob * 30

        # Sharpe score: scaled 0-20 based on Sharpe ratio
        sharpe_score = min(20, max(0, sharpe * 10))

        final_score = ev_score + prob_score + sharpe_score

        explanation = self._build_explanation(
            win_prob, profit_if_win, loss_if_lose, expected_value, kelly
        )

        return ComponentScore(
            name="risk_reward",
            score=max(0, min(100, final_score)),
            weight=self.config.weights.get("risk_reward", 0.10),
            weighted_score=0.0,
            explanation=explanation,
            metadata={
                "is_arbitrage": False,
                "profit_potential": profit_if_win,
                "loss_potential": loss_if_lose,
                "win_probability": win_prob,
                "expected_value": expected_value,
                "kelly_fraction": kelly,
                "sharpe_estimate": sharpe,
            }
        )

    def _adjust_probability(
        self,
        divergence: Divergence,
        base_prob: float
    ) -> float:
        """Adjust win probability based on signal characteristics."""
        prob = base_prob

        # Adjust based on confidence
        confidence = divergence.confidence
        prob = prob * (0.5 + confidence * 0.5)  # Scale by confidence

        # Adjust based on evidence
        evidence_count = len(divergence.supporting_evidence or [])
        if evidence_count >= 3:
            prob = min(0.95, prob * 1.1)
        elif evidence_count == 0:
            prob = prob * 0.9

        # Cap at reasonable bounds
        return max(0.1, min(0.95, prob))

    def _estimate_loss(self, divergence: Divergence) -> float:
        """
        Estimate potential loss if signal is wrong.

        For prediction markets, max loss is typically the price paid.
        """
        prices = divergence.current_prices
        if not prices:
            return 0.5  # Default assumption

        # Use the trade direction to estimate loss
        direction = divergence.direction

        if "BUY" in direction:
            # If buying, loss is the price paid
            # Find the buy price
            if len(prices) == 2:
                # Assume buying the cheaper one
                return min(prices.values())
            return 0.5
        elif "SELL" in direction:
            # If selling, loss is 1 - price (if it goes to 1)
            if len(prices) == 2:
                return 1 - max(prices.values())
            return 0.5
        else:
            # Unknown direction
            return 0.5

    def _calculate_kelly(
        self,
        win_prob: float,
        profit: float,
        loss: float
    ) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly = (bp - q) / b
        where b = profit/loss ratio, p = win prob, q = 1-p
        """
        if loss <= 0:
            return 0

        b = profit / loss if loss > 0 else 0
        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        # Cap and apply fraction
        kelly = max(0, min(1, kelly))
        return kelly * self.kelly_fraction

    def _estimate_sharpe(
        self,
        expected_value: float,
        profit: float,
        loss: float,
        win_prob: float
    ) -> float:
        """
        Estimate Sharpe-like ratio.

        Sharpe = (E[R] - Rf) / std(R)
        """
        # Estimate standard deviation of returns
        mean_return = expected_value

        # Variance = p * (profit - mean)^2 + (1-p) * (-loss - mean)^2
        variance = (
            win_prob * (profit - mean_return) ** 2 +
            (1 - win_prob) * (-loss - mean_return) ** 2
        )
        std = math.sqrt(variance) if variance > 0 else 0.001

        # Annualized (assume daily, ~252 trading days)
        # But for quick trades, just use raw ratio
        sharpe = (mean_return - 0) / std if std > 0 else 0

        return sharpe

    def _score_expected_value(self, ev: float) -> float:
        """Convert expected value to score (0-50 range)."""
        if ev <= 0:
            return max(0, 25 + ev * 500)  # Negative EV gets 0-25
        elif ev < 0.01:
            return 25 + ev * 1500  # 25-40 for 0-1¢ EV
        elif ev < 0.03:
            return 40 + (ev - 0.01) * 250  # 40-45 for 1-3¢
        else:
            return min(50, 45 + (ev - 0.03) * 50)  # 45-50 for 3¢+

    def _build_explanation(
        self,
        win_prob: float,
        profit: float,
        loss: float,
        ev: float,
        kelly: float
    ) -> str:
        """Build human-readable explanation."""
        parts = []

        # Win probability assessment
        if win_prob >= 0.8:
            parts.append(f"High win probability ({win_prob:.0%})")
        elif win_prob >= 0.6:
            parts.append(f"Moderate win probability ({win_prob:.0%})")
        else:
            parts.append(f"Lower win probability ({win_prob:.0%})")

        # Expected value
        if ev > 0:
            parts.append(f"+{ev*100:.1f}¢ expected value")
        elif ev < 0:
            parts.append(f"{ev*100:.1f}¢ expected value (negative)")
        else:
            parts.append("break-even expected value")

        # Risk/reward ratio
        if loss > 0:
            ratio = profit / loss
            if ratio >= 2:
                parts.append(f"favorable {ratio:.1f}:1 reward/risk")
            elif ratio >= 1:
                parts.append(f"acceptable {ratio:.1f}:1 reward/risk")
            else:
                parts.append(f"unfavorable {ratio:.1f}:1 reward/risk")

        # Kelly recommendation
        if kelly >= 0.1:
            parts.append(f"Kelly suggests {kelly:.0%} position")

        return ". ".join(parts)

    def calculate_position_size(
        self,
        divergence: Divergence,
        bankroll: float
    ) -> float:
        """
        Calculate recommended position size.

        Uses Kelly criterion with configured fraction.
        """
        if divergence.is_arbitrage:
            # For arbitrage, limited by liquidity
            max_by_kelly = bankroll * self.config.max_position_fraction
            return min(divergence.max_executable_size, max_by_kelly)

        # Calculate Kelly fraction
        dtype = divergence.divergence_type
        base_prob = self.WIN_PROBABILITIES.get(dtype, 0.5)
        win_prob = self._adjust_probability(divergence, base_prob)

        profit = divergence.profit_potential
        loss = self._estimate_loss(divergence)

        kelly = self._calculate_kelly(win_prob, profit, loss)

        # Position size
        kelly_size = bankroll * kelly

        # Cap at max position
        max_size = bankroll * self.config.max_position_fraction

        # Also cap at available liquidity
        return min(kelly_size, max_size, divergence.max_executable_size)

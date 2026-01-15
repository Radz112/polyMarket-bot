"""
Liquidity assessment for divergence trading.

Analyzes orderbooks to determine executable sizes and effective prices
for potential trades.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.models import Orderbook


@dataclass
class LiquidityAnalysis:
    """Result of analyzing liquidity for a trade."""
    executable_size: float       # How much can be traded within slippage
    effective_price: float       # Volume-weighted average price
    total_available: float       # Total depth in orderbook
    price_impact: float          # Estimated price impact percentage
    levels_consumed: int         # Number of price levels consumed


@dataclass
class TwoSidedLiquidity:
    """Result of analyzing liquidity for a two-sided arbitrage trade."""
    executable_size: float       # Max size tradeable on both sides
    expected_profit: float       # Expected profit at executable size
    profit_per_dollar: float     # Profit per dollar risked
    side_a_analysis: LiquidityAnalysis
    side_b_analysis: LiquidityAnalysis


class LiquidityAssessor:
    """
    Analyzes orderbook liquidity for trade execution.

    Provides methods to:
    - Calculate effective prices for given sizes
    - Determine maximum executable sizes within slippage limits
    - Assess two-sided liquidity for arbitrage opportunities
    """

    def get_effective_price(
        self,
        levels: List[Tuple[float, float]],
        size: float
    ) -> Tuple[float, float]:
        """
        Calculate volume-weighted average price (VWAP) for a given size.

        Args:
            levels: List of (price, size) tuples, sorted appropriately
                   (ascending for asks when buying, descending for bids when selling)
            size: Desired trade size in dollars

        Returns:
            Tuple of (average_price, filled_size)
        """
        if not levels or size <= 0:
            return 0.0, 0.0

        remaining = size
        total_cost = 0.0
        filled = 0.0

        for price, depth in levels:
            take = min(remaining, depth)
            total_cost += take * price
            filled += take
            remaining -= take
            if remaining <= 0:
                break

        if filled == 0:
            return 0.0, 0.0

        return total_cost / filled, filled

    def get_executable_size(
        self,
        orderbook: Orderbook,
        side: str,
        max_slippage: float = 0.01
    ) -> float:
        """
        Determine how much can be traded before moving price by max_slippage.

        Args:
            orderbook: The current orderbook
            side: "buy" (hit asks) or "sell" (hit bids)
            max_slippage: Maximum acceptable price movement (e.g., 0.01 = 1%)

        Returns:
            Maximum executable size in dollars
        """
        if side == "buy":
            levels = orderbook.asks
            if not levels:
                return 0.0
            best_price = float(levels[0].price)
            limit_price = best_price * (1 + max_slippage)
        else:
            levels = orderbook.bids
            if not levels:
                return 0.0
            best_price = float(levels[0].price)
            limit_price = best_price * (1 - max_slippage)

        max_size = 0.0
        for lvl in levels:
            p = float(lvl.price)
            s = float(lvl.size)

            if side == "buy":
                if p > limit_price:
                    break
            else:
                if p < limit_price:
                    break

            max_size += s

        return max_size

    def get_max_executable_size(
        self,
        orderbook: Orderbook,
        side: str,
        max_slippage: float = 0.01
    ) -> float:
        """Alias for get_executable_size for backward compatibility."""
        return self.get_executable_size(orderbook, side, max_slippage)

    def analyze_liquidity(
        self,
        orderbook: Orderbook,
        side: str,
        max_slippage: float = 0.01
    ) -> LiquidityAnalysis:
        """
        Comprehensive liquidity analysis for one side of the orderbook.

        Args:
            orderbook: The current orderbook
            side: "buy" or "sell"
            max_slippage: Maximum acceptable slippage

        Returns:
            LiquidityAnalysis with detailed breakdown
        """
        if side == "buy":
            levels = orderbook.asks
        else:
            levels = orderbook.bids

        if not levels:
            return LiquidityAnalysis(
                executable_size=0.0,
                effective_price=0.0,
                total_available=0.0,
                price_impact=0.0,
                levels_consumed=0
            )

        best_price = float(levels[0].price)

        if side == "buy":
            limit_price = best_price * (1 + max_slippage)
        else:
            limit_price = best_price * (1 - max_slippage)

        executable_size = 0.0
        total_cost = 0.0
        levels_consumed = 0
        total_available = sum(float(lvl.size) for lvl in levels)

        for lvl in levels:
            p = float(lvl.price)
            s = float(lvl.size)

            if side == "buy":
                if p > limit_price:
                    break
            else:
                if p < limit_price:
                    break

            executable_size += s
            total_cost += s * p
            levels_consumed += 1

        if executable_size > 0:
            effective_price = total_cost / executable_size
            price_impact = abs(effective_price - best_price) / best_price
        else:
            effective_price = best_price
            price_impact = 0.0

        return LiquidityAnalysis(
            executable_size=executable_size,
            effective_price=effective_price,
            total_available=total_available,
            price_impact=price_impact,
            levels_consumed=levels_consumed
        )

    def assess_two_sided_liquidity(
        self,
        orderbook_a: Orderbook,
        side_a: str,
        orderbook_b: Orderbook,
        side_b: str,
        max_slippage: float = 0.01
    ) -> TwoSidedLiquidity:
        """
        Assess liquidity for a two-sided arbitrage trade.

        For arbitrage, we need to execute on both sides simultaneously.
        This determines the maximum size that can be traded on both sides
        and the expected profit.

        Args:
            orderbook_a: Orderbook for first market
            side_a: "buy" or "sell" for first market
            orderbook_b: Orderbook for second market
            side_b: "buy" or "sell" for second market
            max_slippage: Maximum acceptable slippage per side

        Returns:
            TwoSidedLiquidity analysis
        """
        # Analyze each side
        analysis_a = self.analyze_liquidity(orderbook_a, side_a, max_slippage)
        analysis_b = self.analyze_liquidity(orderbook_b, side_b, max_slippage)

        # Maximum executable size is limited by the smaller side
        executable_size = min(analysis_a.executable_size, analysis_b.executable_size)

        if executable_size <= 0:
            return TwoSidedLiquidity(
                executable_size=0.0,
                expected_profit=0.0,
                profit_per_dollar=0.0,
                side_a_analysis=analysis_a,
                side_b_analysis=analysis_b
            )

        # Calculate effective prices for the executable size
        levels_a = [(float(l.price), float(l.size)) for l in
                    (orderbook_a.asks if side_a == "buy" else orderbook_a.bids)]
        levels_b = [(float(l.price), float(l.size)) for l in
                    (orderbook_b.asks if side_b == "buy" else orderbook_b.bids)]

        price_a, _ = self.get_effective_price(levels_a, executable_size)
        price_b, _ = self.get_effective_price(levels_b, executable_size)

        # Calculate profit based on trade direction
        # If buying A and selling B: profit = (sell_price - buy_price) * size
        # If selling A and buying B: profit = (sell_price - buy_price) * size
        if side_a == "buy" and side_b == "sell":
            # Buy A, sell B
            expected_profit = (price_b - price_a) * executable_size
        elif side_a == "sell" and side_b == "buy":
            # Sell A, buy B
            expected_profit = (price_a - price_b) * executable_size
        else:
            # Same side on both (unusual for arbitrage)
            expected_profit = abs(price_a - price_b) * executable_size

        # Profit per dollar: assuming we risk the buy price * size
        if side_a == "buy":
            risk = price_a * executable_size
        else:
            risk = price_b * executable_size

        profit_per_dollar = expected_profit / risk if risk > 0 else 0.0

        return TwoSidedLiquidity(
            executable_size=executable_size,
            expected_profit=expected_profit,
            profit_per_dollar=profit_per_dollar,
            side_a_analysis=analysis_a,
            side_b_analysis=analysis_b
        )

    def estimate_execution_cost(
        self,
        orderbook: Orderbook,
        side: str,
        size: float
    ) -> Tuple[float, float, float]:
        """
        Estimate the cost of executing a trade of given size.

        Args:
            orderbook: The orderbook to analyze
            side: "buy" or "sell"
            size: Trade size in dollars

        Returns:
            Tuple of (effective_price, slippage_cost, total_cost)
            - effective_price: VWAP for the trade
            - slippage_cost: Additional cost due to walking the book
            - total_cost: Total execution cost
        """
        if side == "buy":
            levels = [(float(l.price), float(l.size)) for l in orderbook.asks]
            best_price = orderbook.best_ask
        else:
            levels = [(float(l.price), float(l.size)) for l in orderbook.bids]
            best_price = orderbook.best_bid

        if not levels or best_price is None:
            return 0.0, 0.0, 0.0

        effective_price, filled = self.get_effective_price(levels, size)

        if filled == 0:
            return 0.0, 0.0, 0.0

        # Slippage cost is the difference from best price
        if side == "buy":
            slippage_cost = (effective_price - best_price) * filled
        else:
            slippage_cost = (best_price - effective_price) * filled

        total_cost = effective_price * filled

        return effective_price, slippage_cost, total_cost

    def calculate_arbitrage_profit(
        self,
        buy_orderbook: Orderbook,
        sell_orderbook: Orderbook,
        size: float
    ) -> Tuple[float, float, float]:
        """
        Calculate profit from buying on one orderbook and selling on another.

        Args:
            buy_orderbook: Orderbook to buy from (hit asks)
            sell_orderbook: Orderbook to sell to (hit bids)
            size: Trade size

        Returns:
            Tuple of (gross_profit, net_profit_after_costs, roi)
        """
        # Get execution prices
        buy_levels = [(float(l.price), float(l.size)) for l in buy_orderbook.asks]
        sell_levels = [(float(l.price), float(l.size)) for l in sell_orderbook.bids]

        buy_price, buy_filled = self.get_effective_price(buy_levels, size)
        sell_price, sell_filled = self.get_effective_price(sell_levels, size)

        # Use the smaller of the two fills
        actual_size = min(buy_filled, sell_filled)

        if actual_size <= 0:
            return 0.0, 0.0, 0.0

        # Recalculate for actual size
        buy_price, _ = self.get_effective_price(buy_levels, actual_size)
        sell_price, _ = self.get_effective_price(sell_levels, actual_size)

        buy_cost = buy_price * actual_size
        sell_revenue = sell_price * actual_size

        gross_profit = sell_revenue - buy_cost

        # Assume some transaction costs (e.g., 0.1% per side)
        tx_cost_rate = 0.001
        tx_costs = (buy_cost + sell_revenue) * tx_cost_rate

        net_profit = gross_profit - tx_costs

        roi = net_profit / buy_cost if buy_cost > 0 else 0.0

        return gross_profit, net_profit, roi

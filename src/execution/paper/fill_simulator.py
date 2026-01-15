"""
Fill simulation for paper trading.

Simulates order fills against an orderbook with configurable slippage models.
"""
import logging
from enum import Enum
from datetime import datetime
from typing import List, Tuple, Optional

from src.models.orderbook import Orderbook, OrderbookEntry
from src.execution.paper.types import FillResult

logger = logging.getLogger(__name__)


class SlippageModel(str, Enum):
    """Slippage simulation models."""
    OPTIMISTIC = "optimistic"    # Fill at best price
    REALISTIC = "realistic"      # Walk the book, account for slippage
    PESSIMISTIC = "pessimistic"  # Assume minimum slippage penalty


class FillSimulator:
    """
    Simulates order fills against an orderbook.
    
    Supports multiple slippage models for different simulation scenarios:
    - optimistic: Always fill at best available price
    - realistic: Walk the orderbook, accounting for size-based slippage
    - pessimistic: Assume worst-case slippage (1% minimum)
    """
    
    # Minimum slippage for pessimistic model
    MIN_PESSIMISTIC_SLIPPAGE = 0.01
    
    def __init__(
        self,
        slippage_model: str = "realistic",
        fee_pct: float = 0.02
    ):
        """
        Initialize the fill simulator.
        
        Args:
            slippage_model: One of "optimistic", "realistic", or "pessimistic"
            fee_pct: Fee percentage to apply to trades (e.g., 0.02 for 2%)
        """
        self.slippage_model = SlippageModel(slippage_model)
        self.fee_pct = fee_pct
    
    def simulate_fill(
        self,
        orderbook: Orderbook,
        side: str,  # "buy" or "sell"
        size: float,
        limit_price: Optional[float] = None
    ) -> FillResult:
        """
        Simulate an order fill against an orderbook.
        
        Args:
            orderbook: Current orderbook snapshot
            side: "buy" or "sell"
            size: Amount to buy/sell
            limit_price: Optional limit price (None for market order)
            
        Returns:
            FillResult with fill details
        """
        if size <= 0:
            return FillResult(
                filled_size=0,
                average_price=0,
                slippage=0,
                fees=0,
                total_cost=0,
                unfilled_size=size,
                fill_time=datetime.utcnow()
            )
        
        # Select orderbook side (buy = hit asks, sell = hit bids)
        if side.lower() == "buy":
            levels = orderbook.asks
            best_price = orderbook.best_ask
        else:
            levels = orderbook.bids
            best_price = orderbook.best_bid
        
        if not levels or best_price is None:
            logger.warning(f"No liquidity on {side} side for {orderbook.market_id}")
            return FillResult(
                filled_size=0,
                average_price=0,
                slippage=0,
                fees=0,
                total_cost=0,
                unfilled_size=size,
                fill_time=datetime.utcnow()
            )
        
        # Calculate fill based on slippage model
        if self.slippage_model == SlippageModel.OPTIMISTIC:
            # Fill at best price
            filled_size, average_price = self._optimistic_fill(best_price, size, limit_price, side)
        elif self.slippage_model == SlippageModel.REALISTIC:
            # Walk the book
            filled_size, average_price = self._realistic_fill(levels, size, limit_price, side)
        else:
            # Pessimistic: apply minimum slippage
            filled_size, average_price = self._pessimistic_fill(best_price, size, limit_price, side)
        
        # Calculate slippage
        slippage = abs(average_price - best_price) if filled_size > 0 else 0
        
        # Calculate fees
        notional = filled_size * average_price
        fees = notional * self.fee_pct
        
        # Calculate total cost
        if side.lower() == "buy":
            total_cost = notional + fees
        else:
            total_cost = notional - fees  # Net proceeds after fees
        
        return FillResult(
            filled_size=filled_size,
            average_price=average_price,
            slippage=slippage,
            fees=fees,
            total_cost=total_cost,
            unfilled_size=size - filled_size,
            fill_time=datetime.utcnow()
        )
    
    def _optimistic_fill(
        self,
        best_price: float,
        size: float,
        limit_price: Optional[float],
        side: str
    ) -> Tuple[float, float]:
        """Fill at best price, respecting limit."""
        if limit_price is not None:
            if side.lower() == "buy" and best_price > limit_price:
                return 0, 0
            if side.lower() == "sell" and best_price < limit_price:
                return 0, 0
        return size, best_price
    
    def _realistic_fill(
        self,
        levels: List[OrderbookEntry],
        size: float,
        limit_price: Optional[float],
        side: str
    ) -> Tuple[float, float]:
        """Walk the orderbook to calculate realistic fill."""
        # Sort levels appropriately
        if side.lower() == "buy":
            # For buys, sort asks ascending (best first)
            sorted_levels = sorted(levels, key=lambda x: x.price)
        else:
            # For sells, sort bids descending (best first)
            sorted_levels = sorted(levels, key=lambda x: x.price, reverse=True)
        
        return self.walk_orderbook(sorted_levels, size, limit_price, side)
    
    def _pessimistic_fill(
        self,
        best_price: float,
        size: float,
        limit_price: Optional[float],
        side: str
    ) -> Tuple[float, float]:
        """Fill with minimum slippage penalty."""
        # Apply slippage in the adverse direction
        if side.lower() == "buy":
            slipped_price = best_price * (1 + self.MIN_PESSIMISTIC_SLIPPAGE)
        else:
            slipped_price = best_price * (1 - self.MIN_PESSIMISTIC_SLIPPAGE)
        
        # Check limit
        if limit_price is not None:
            if side.lower() == "buy" and slipped_price > limit_price:
                return 0, 0
            if side.lower() == "sell" and slipped_price < limit_price:
                return 0, 0
        
        return size, slipped_price
    
    def walk_orderbook(
        self,
        levels: List[OrderbookEntry],
        size: float,
        limit_price: Optional[float] = None,
        side: str = "buy"
    ) -> Tuple[float, float]:
        """
        Walk through orderbook levels to simulate a fill.
        
        Args:
            levels: Sorted orderbook levels (best price first)
            size: Amount to fill
            limit_price: Optional limit price
            side: "buy" or "sell" (for limit price comparison)
            
        Returns:
            Tuple of (filled_size, average_price)
        """
        remaining = size
        total_cost = 0.0
        filled = 0.0
        
        for level in levels:
            # Check limit price
            if limit_price is not None:
                if side.lower() == "buy" and level.price > limit_price:
                    break
                if side.lower() == "sell" and level.price < limit_price:
                    break
            
            # Fill from this level
            fill_amount = min(remaining, level.size)
            total_cost += fill_amount * level.price
            filled += fill_amount
            remaining -= fill_amount
            
            if remaining <= 0:
                break
        
        if filled == 0:
            return 0, 0
        
        average_price = total_cost / filled
        return filled, average_price
    
    def estimate_slippage(
        self,
        orderbook: Orderbook,
        side: str,
        size: float
    ) -> float:
        """
        Estimate slippage for a given order size.
        
        Args:
            orderbook: Current orderbook snapshot
            side: "buy" or "sell"
            size: Order size
            
        Returns:
            Estimated slippage (price difference from best)
        """
        if side.lower() == "buy":
            levels = orderbook.asks
            best_price = orderbook.best_ask
        else:
            levels = orderbook.bids
            best_price = orderbook.best_bid
        
        if not levels or best_price is None:
            return float('inf')
        
        # Simulate fill to get average price
        sorted_levels = sorted(
            levels,
            key=lambda x: x.price,
            reverse=(side.lower() == "sell")
        )
        
        _, avg_price = self.walk_orderbook(sorted_levels, size, None, side)
        
        if avg_price == 0:
            return float('inf')
        
        return abs(avg_price - best_price)
    
    def can_fill(
        self,
        orderbook: Orderbook,
        side: str,
        size: float,
        limit_price: Optional[float] = None
    ) -> bool:
        """
        Check if an order can be fully filled.
        
        Args:
            orderbook: Current orderbook snapshot
            side: "buy" or "sell"
            size: Order size
            limit_price: Optional limit price
            
        Returns:
            True if order can be fully filled
        """
        result = self.simulate_fill(orderbook, side, size, limit_price)
        return result.unfilled_size == 0

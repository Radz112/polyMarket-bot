"""
Smart order routing for optimal execution.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List

from src.config.settings import Config
from src.models.orderbook import Orderbook
from src.execution.orders.models import Order, OrderType, ExecutionPlan

logger = logging.getLogger(__name__)


class SmartOrderRouter:
    """
    Plans optimal order execution strategy.
    
    Features:
    - Order splitting for large orders
    - Urgency-based limit price calculation
    - Fill probability estimation
    """
    
    # Thresholds
    SPLIT_THRESHOLD_PCT = 0.30  # Split if order > 30% of top level
    MAX_SLIPPAGE_PCT = 0.02    # 2% max slippage for immediate
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize router.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def plan_execution(
        self,
        order: Order,
        orderbook: Orderbook
    ) -> ExecutionPlan:
        """
        Plan optimal execution strategy.
        
        Args:
            order: Order to plan
            orderbook: Current orderbook
            
        Returns:
            ExecutionPlan with strategy and child orders
        """
        # Check if we should split
        if self.should_split_order(order, orderbook):
            return self._plan_split_execution(order, orderbook)
        
        # Single order execution
        estimated_price = self._estimate_fill_price(order, orderbook)
        best_price = self._get_best_price(order, orderbook)
        slippage = abs(estimated_price - best_price) if best_price else 0
        
        return ExecutionPlan(
            original_order=order,
            child_orders=[],
            strategy="single",
            estimated_fill_price=estimated_price,
            estimated_slippage=slippage,
            estimated_time_to_fill=self._estimate_fill_time(order, orderbook)
        )
    
    def should_split_order(
        self,
        order: Order,
        orderbook: Orderbook
    ) -> bool:
        """
        Determine if order should be split.
        
        Split if:
        - Order exceeds threshold of top level liquidity
        - Better prices available at multiple levels
        
        Args:
            order: Order to check
            orderbook: Current orderbook
            
        Returns:
            True if order should be split
        """
        # Get relevant side of book
        if order.action == "BUY":
            levels = orderbook.asks
        else:
            levels = orderbook.bids
        
        if not levels:
            return False
        
        # Check against top level
        top_level_size = levels[0].size
        
        if order.size > top_level_size * self.SPLIT_THRESHOLD_PCT:
            # Check if there are multiple levels
            if len(levels) > 1:
                return True
        
        return False
    
    def calculate_optimal_limit_price(
        self,
        order: Order,
        orderbook: Orderbook,
        urgency: str = "normal"
    ) -> float:
        """
        Calculate optimal limit price based on urgency.
        
        Args:
            order: Order to price
            orderbook: Current orderbook
            urgency: "immediate", "normal", or "patient"
            
        Returns:
            Optimal limit price
        """
        best_bid = orderbook.best_bid or 0.45
        best_ask = orderbook.best_ask or 0.55
        spread = best_ask - best_bid
        mid = orderbook.mid_price or (best_bid + best_ask) / 2
        
        if order.action == "BUY":
            if urgency == "immediate":
                # Cross spread - pay ask
                return min(best_ask * 1.005, 0.99)  # Slightly above best ask
            elif urgency == "patient":
                # Inside spread
                return best_bid + spread * 0.25
            else:  # normal
                # At best bid
                return best_bid
        else:  # SELL
            if urgency == "immediate":
                # Cross spread - hit bid
                return max(best_bid * 0.995, 0.01)  # Slightly below best bid
            elif urgency == "patient":
                # Inside spread
                return best_ask - spread * 0.25
            else:  # normal
                # At best ask
                return best_ask
    
    def estimate_fill_probability(
        self,
        order: Order,
        orderbook: Orderbook
    ) -> float:
        """
        Estimate probability of fill at given price.
        
        Args:
            order: Order with limit price set
            orderbook: Current orderbook
            
        Returns:
            Probability [0, 1]
        """
        if order.order_type == OrderType.MARKET:
            # Market orders always fill (if liquidity exists)
            total_liquidity = sum(
                level.size for level in 
                (orderbook.asks if order.action == "BUY" else orderbook.bids)
            )
            if total_liquidity >= order.size:
                return 1.0
            elif total_liquidity > 0:
                return total_liquidity / order.size
            return 0.0
        
        if order.limit_price is None:
            return 0.0
        
        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        
        if order.action == "BUY":
            if best_ask is None:
                return 0.0
            if order.limit_price >= best_ask:
                return 0.95  # Almost certain
            elif order.limit_price >= best_bid:
                # In spread - good chance
                spread = best_ask - best_bid
                position = (order.limit_price - best_bid) / spread if spread > 0 else 0.5
                return 0.3 + position * 0.5
            else:
                # Below best bid - unlikely
                return 0.1
        else:  # SELL
            if best_bid is None:
                return 0.0
            if order.limit_price <= best_bid:
                return 0.95
            elif order.limit_price <= best_ask:
                spread = best_ask - best_bid
                position = (best_ask - order.limit_price) / spread if spread > 0 else 0.5
                return 0.3 + position * 0.5
            else:
                return 0.1
    
    def _plan_split_execution(
        self,
        order: Order,
        orderbook: Orderbook
    ) -> ExecutionPlan:
        """Plan execution with order splitting."""
        # Get relevant side of book
        if order.action == "BUY":
            levels = orderbook.asks
        else:
            levels = orderbook.bids
        
        child_orders = []
        remaining = order.size
        total_value = 0
        
        for level in levels:
            if remaining <= 0:
                break
            
            # Take portion of this level
            size_at_level = min(remaining, level.size * 0.8)  # Leave 20% for others
            
            if size_at_level > 0:
                child = Order(
                    id=f"{order.id}_child_{len(child_orders)}",
                    market_id=order.market_id,
                    market_name=order.market_name,
                    side=order.side,
                    action=order.action,
                    order_type=OrderType.LIMIT,
                    size=size_at_level,
                    limit_price=level.price,
                    parent_order_id=order.id,
                    signal_id=order.signal_id
                )
                child_orders.append(child)
                total_value += size_at_level * level.price
                remaining -= size_at_level
        
        # Calculate estimated fill price
        total_size = sum(c.size for c in child_orders)
        estimated_fill_price = total_value / total_size if total_size > 0 else 0
        
        # Slippage from best price
        best_price = self._get_best_price(order, orderbook)
        slippage = abs(estimated_fill_price - best_price) if best_price else 0
        
        return ExecutionPlan(
            original_order=order,
            child_orders=child_orders,
            strategy="split",
            estimated_fill_price=estimated_fill_price,
            estimated_slippage=slippage,
            estimated_time_to_fill=timedelta(seconds=len(child_orders) * 2)
        )
    
    def _estimate_fill_price(self, order: Order, orderbook: Orderbook) -> float:
        """Estimate fill price for order."""
        if order.order_type == OrderType.LIMIT and order.limit_price:
            return order.limit_price
        
        # Walk the book for market orders
        if order.action == "BUY":
            levels = orderbook.asks
        else:
            levels = orderbook.bids
        
        if not levels:
            return 0.50
        
        remaining = order.size
        total_value = 0
        
        for level in levels:
            if remaining <= 0:
                break
            
            fill_size = min(remaining, level.size)
            total_value += fill_size * level.price
            remaining -= fill_size
        
        filled_size = order.size - remaining
        if filled_size > 0:
            return total_value / filled_size
        
        return levels[0].price
    
    def _get_best_price(self, order: Order, orderbook: Orderbook) -> Optional[float]:
        """Get best available price for order."""
        if order.action == "BUY":
            return orderbook.best_ask
        else:
            return orderbook.best_bid
    
    def _estimate_fill_time(
        self,
        order: Order,
        orderbook: Orderbook
    ) -> timedelta:
        """Estimate time to fill order."""
        if order.order_type == OrderType.MARKET:
            return timedelta(seconds=1)
        
        fill_prob = self.estimate_fill_probability(order, orderbook)
        
        if fill_prob > 0.9:
            return timedelta(seconds=5)
        elif fill_prob > 0.5:
            return timedelta(minutes=5)
        elif fill_prob > 0.2:
            return timedelta(hours=1)
        else:
            return timedelta(hours=24)

"""
Order management for trade execution.
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, List

from src.config.settings import Config
from src.execution.orders.models import (
    Order,
    OrderType,
    OrderStatus,
    OrderResult,
    ExecutionPlan,
)
from src.execution.orders.validator import OrderValidator
from src.execution.orders.executor import OrderExecutor
from src.execution.orders.router import SmartOrderRouter
from src.execution.orders.tracker import OrderTracker
from src.execution.positions import PositionManager

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Main interface for order management.
    
    Orchestrates:
    - Order validation
    - Smart routing
    - Execution
    - Tracking
    """
    
    def __init__(
        self,
        config: Config,
        position_manager: Optional[PositionManager] = None,
        executor: Optional[OrderExecutor] = None,
        is_paper: bool = True,
        api_client = None,  # LiveTradingClient for live trading
    ):
        """
        Initialize order manager.
        
        Args:
            config: Application configuration
            position_manager: For position-aware validation
            executor: Order executor (paper or live)
            is_paper: Whether in paper trading mode
            api_client: LiveTradingClient for live order execution
        """
        self.config = config
        self.is_paper = is_paper
        
        # Components
        self.position_manager = position_manager
        self.validator = OrderValidator(config, position_manager)
        self.executor = executor or OrderExecutor(
            is_paper=is_paper,
            api_client=api_client,
        )
        self.router = SmartOrderRouter(config)
        self.tracker = OrderTracker()
        
        # State
        self.balance = config.initial_capital
    
    async def submit_order(
        self,
        order: Order,
        orderbook=None
    ) -> OrderResult:
        """
        Submit order for execution.
        
        1. Validate order
        2. Check risk limits
        3. Plan execution
        4. Execute
        5. Track
        
        Args:
            order: Order to submit
            orderbook: Current orderbook for routing
            
        Returns:
            OrderResult with execution details
        """
        # Validate
        validation = self.validator.validate(order, self.balance)
        
        if not validation.is_valid:
            order.reject("; ".join(validation.errors))
            return OrderResult(
                success=False,
                order=order,
                message="Order validation failed",
                error="; ".join(validation.errors)
            )
        
        # Log warnings
        for warning in validation.warnings:
            logger.warning(f"Order {order.id}: {warning}")
        
        # Plan execution
        if orderbook:
            plan = self.router.plan_execution(order, orderbook)
            
            if plan.strategy == "split" and plan.child_orders:
                return await self._execute_split_order(plan, orderbook)
        
        # Execute single order
        result = await self.executor.execute(order, orderbook)
        
        # Track
        await self.tracker.track_order(order)
        
        # Update balance for paper trading
        if result.success and self.is_paper and order.action == "BUY":
            cost = order.filled_size * (order.average_fill_price or order.limit_price or 0.5)
            self.balance -= cost + order.actual_fees
        elif result.success and self.is_paper and order.action == "SELL":
            proceeds = order.filled_size * (order.average_fill_price or order.limit_price or 0.5)
            self.balance += proceeds - order.actual_fees
        
        return result
    
    async def submit_signal_order(
        self,
        signal,  # ScoredSignal
        size_override: float = None,
        orderbook=None
    ) -> OrderResult:
        """
        Convert signal to order and submit.
        
        Args:
            signal: ScoredSignal with recommended action
            size_override: Override signal's size recommendation
            orderbook: Current orderbook
            
        Returns:
            OrderResult
        """
        # Extract signal parameters
        market_id = signal.divergence.market_ids[0] if signal.divergence.market_ids else None
        if not market_id:
            return OrderResult(
                success=False,
                order=None,
                message="Signal has no market ID",
                error="Invalid signal"
            )
        
        # Determine action and side
        action = signal.recommended_action  # "BUY" or "SELL"
        side = "YES"  # Default, could be derived from signal
        
        # Get size and price
        size = size_override or signal.recommended_size
        price = signal.recommended_price
        
        # Create order
        order = Order(
            id=f"order_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            market_name=market_id,  # Could look up name
            side=side,
            action=action,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            size=size,
            limit_price=price,
            signal_id=signal.divergence.id if hasattr(signal.divergence, 'id') else None,
            signal_score=signal.overall_score
        )
        
        return await self.submit_order(order, orderbook)
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.
        
        Args:
            order_id: Order to cancel
            
        Returns:
            True if cancelled
        """
        order = self.tracker.get_order(order_id)
        if order is None:
            logger.warning(f"Order {order_id} not found")
            return False
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return False
        
        success = await self.executor.cancel_order(order)
        if success:
            await self.tracker.cancel_order(order_id, "Cancelled by user")
        
        return success
    
    async def cancel_all_orders(self, market_id: str = None) -> int:
        """
        Cancel all pending orders.
        
        Args:
            market_id: Optional filter by market
            
        Returns:
            Number of orders cancelled
        """
        orders = self.tracker.get_active_orders(market_id)
        cancelled = 0
        
        for order in orders:
            if await self.cancel_order(order.id):
                cancelled += 1
        
        return cancelled
    
    async def modify_order(
        self,
        order_id: str,
        new_price: float = None,
        new_size: float = None
    ) -> OrderResult:
        """
        Modify existing order.
        
        Args:
            order_id: Order to modify
            new_price: New limit price
            new_size: New size
            
        Returns:
            OrderResult
        """
        order = self.tracker.get_order(order_id)
        if order is None:
            return OrderResult(
                success=False,
                order=None,
                message=f"Order {order_id} not found",
                error="Order not found"
            )
        
        if not order.is_active:
            return OrderResult(
                success=False,
                order=order,
                message="Order is not active",
                error="Cannot modify inactive order"
            )
        
        # Cancel and resubmit with new params
        await self.cancel_order(order_id)
        
        # Create new order with modifications
        new_order = Order(
            id=f"order_{uuid.uuid4().hex[:8]}",
            market_id=order.market_id,
            market_name=order.market_name,
            side=order.side,
            action=order.action,
            order_type=order.order_type,
            size=new_size or order.remaining_size,
            limit_price=new_price or order.limit_price,
            signal_id=order.signal_id,
            parent_order_id=order_id
        )
        
        return await self.submit_order(new_order)
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current order status.
        
        Args:
            order_id: Order to check
            
        Returns:
            Order or None
        """
        return self.tracker.get_order(order_id)
    
    async def sync_orders(self) -> int:
        """
        Sync local order state with exchange.
        
        Returns:
            Number of orders synced
        """
        # For paper trading, nothing to sync
        if self.is_paper:
            return 0
        
        # For live, would query exchange for order status
        # and update local state
        # TODO: Implement live sync
        return 0
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return self.tracker.get_pending_orders()
    
    def get_active_orders(self, market_id: str = None) -> List[Order]:
        """Get all active orders."""
        return self.tracker.get_active_orders(market_id)
    
    async def _execute_split_order(
        self,
        plan: ExecutionPlan,
        orderbook
    ) -> OrderResult:
        """Execute a split order plan."""
        results = []
        
        for child in plan.child_orders:
            result = await self.executor.execute(child, orderbook)
            await self.tracker.track_order(child)
            results.append(result)
        
        # Aggregate results
        successful = [r for r in results if r.success]
        total_filled = sum(r.order.filled_size for r in successful)
        
        # Update parent order
        parent = plan.original_order
        if total_filled > 0:
            parent.filled_size = total_filled
            parent.status = OrderStatus.FILLED if total_filled >= parent.size else OrderStatus.PARTIAL
        
        await self.tracker.track_order(parent)
        
        return OrderResult(
            success=len(successful) > 0,
            order=parent,
            message=f"Split order: {len(successful)}/{len(results)} child orders successful"
        )
    
    def create_order(
        self,
        market_id: str,
        side: str,
        action: str,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float = None,
        signal_id: str = None
    ) -> Order:
        """
        Create a new order object.
        
        Convenience method for creating orders.
        """
        return Order(
            id=f"order_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            market_name=market_id,
            side=side,
            action=action,
            order_type=order_type,
            size=size,
            limit_price=limit_price,
            signal_id=signal_id,
            estimated_fees=size * (limit_price or 0.5) * self.config.trading_fees_pct
        )

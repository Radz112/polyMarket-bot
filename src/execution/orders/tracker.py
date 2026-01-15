"""
Order tracking and state management.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable

from src.database.postgres import DatabaseManager
from src.database.redis_cache import CacheManager
from src.execution.orders.models import Order, OrderStatus, Fill

logger = logging.getLogger(__name__)


class OrderTracker:
    """
    Tracks orders and their state.
    
    Features:
    - Active order management
    - Fill event handling
    - Order history
    - Update callbacks
    """
    
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        cache: Optional[CacheManager] = None
    ):
        """
        Initialize tracker.
        
        Args:
            db: Database for persistence
            cache: Cache for real-time state
        """
        self.db = db
        self.cache = cache
        
        # In-memory order storage
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []
        
        # Callbacks for order updates
        self._callbacks: List[Callable[[Order], None]] = []
    
    async def track_order(self, order: Order) -> None:
        """
        Add order to tracking.
        
        Args:
            order: Order to track
        """
        self.active_orders[order.id] = order
        logger.debug(f"Tracking order {order.id}")
        
        # Notify callbacks
        await self._notify_update(order)
    
    async def update_order(self, order_id: str, updates: dict) -> Optional[Order]:
        """
        Update order state.
        
        Args:
            order_id: Order to update
            updates: Dict of field updates
            
        Returns:
            Updated order or None
        """
        order = self.active_orders.get(order_id)
        if order is None:
            logger.warning(f"Order {order_id} not found for update")
            return None
        
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        # Check if order is complete
        if order.is_complete:
            await self._move_to_completed(order)
        
        # Notify callbacks
        await self._notify_update(order)
        
        return order
    
    async def on_fill(self, order_id: str, fill: Fill) -> Optional[Order]:
        """
        Handle order fill event.
        
        Args:
            order_id: Order that was filled
            fill: Fill details
            
        Returns:
            Updated order or None
        """
        order = self.active_orders.get(order_id)
        if order is None:
            logger.warning(f"Order {order_id} not found for fill")
            return None
        
        order.add_fill(fill)
        
        logger.info(
            f"Order {order_id} fill: {fill.size} @ {fill.price:.4f} "
            f"({order.fill_rate:.1f}% filled)"
        )
        
        # Check if order is complete
        if order.is_complete:
            await self._move_to_completed(order)
        
        # Notify callbacks
        await self._notify_update(order)
        
        return order
    
    async def cancel_order(self, order_id: str, reason: str = None) -> bool:
        """
        Cancel a tracked order.
        
        Args:
            order_id: Order to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancelled
        """
        order = self.active_orders.get(order_id)
        if order is None:
            return False
        
        order.cancel(reason)
        await self._move_to_completed(order)
        await self._notify_update(order)
        
        logger.info(f"Order {order_id} cancelled: {reason}")
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID (active or completed)."""
        order = self.active_orders.get(order_id)
        if order:
            return order
        
        for completed in self.completed_orders:
            if completed.id == order_id:
                return completed
        
        return None
    
    def get_active_orders(self, market_id: str = None) -> List[Order]:
        """
        Get all active orders.
        
        Args:
            market_id: Optional filter by market
            
        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())
        
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        
        return orders
    
    def get_pending_orders(self) -> List[Order]:
        """Get orders in PENDING status."""
        return [
            o for o in self.active_orders.values()
            if o.status == OrderStatus.PENDING
        ]
    
    def get_submitted_orders(self) -> List[Order]:
        """Get orders in SUBMITTED or PARTIAL status."""
        return [
            o for o in self.active_orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
        ]
    
    async def get_order_history(
        self,
        start: datetime = None,
        market_id: str = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get historical orders.
        
        Args:
            start: Filter by start time
            market_id: Filter by market
            limit: Max orders to return
            
        Returns:
            List of completed orders
        """
        orders = self.completed_orders.copy()
        
        if start:
            orders = [o for o in orders if o.created_at >= start]
        
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        
        # Sort by created time, newest first
        orders.sort(key=lambda o: o.created_at, reverse=True)
        
        return orders[:limit]
    
    def on_order_update(self, callback: Callable[[Order], None]) -> None:
        """
        Register callback for order updates.
        
        Args:
            callback: Function to call on order updates
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Order], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def _notify_update(self, order: Order) -> None:
        """Notify all callbacks of order update."""
        for callback in self._callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")
    
    async def _move_to_completed(self, order: Order) -> None:
        """Move order from active to completed."""
        if order.id in self.active_orders:
            del self.active_orders[order.id]
            self.completed_orders.append(order)
    
    def get_stats(self) -> dict:
        """Get order tracking statistics."""
        active = list(self.active_orders.values())
        completed = self.completed_orders
        
        filled = [o for o in completed if o.status == OrderStatus.FILLED]
        cancelled = [o for o in completed if o.status == OrderStatus.CANCELLED]
        rejected = [o for o in completed if o.status == OrderStatus.REJECTED]
        
        return {
            "active_count": len(active),
            "completed_count": len(completed),
            "filled_count": len(filled),
            "cancelled_count": len(cancelled),
            "rejected_count": len(rejected),
            "fill_rate": len(filled) / len(completed) if completed else 0,
            "total_volume": sum(o.filled_size for o in filled),
            "total_fees": sum(o.actual_fees for o in filled),
        }
    
    def clear(self) -> None:
        """Clear all tracked orders."""
        self.active_orders.clear()
        self.completed_orders.clear()

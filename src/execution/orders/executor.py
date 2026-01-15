"""
Order execution for paper and live trading.
"""
import logging
from datetime import datetime
from typing import Optional

from src.execution.orders.models import (
    Order,
    OrderType,
    OrderStatus,
    OrderResult,
    Fill,
)
from src.execution.paper import FillSimulator, PaperTrader

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Executes orders against paper or live exchange.
    
    Provides clean abstraction for:
    - Paper trading (simulated fills)
    - Live trading (via API)
    """
    
    def __init__(
        self,
        paper_trader: Optional[PaperTrader] = None,
        fill_simulator: Optional[FillSimulator] = None,
        api_client = None,  # ClobClient for live
        is_paper: bool = True
    ):
        """
        Initialize executor.
        
        Args:
            paper_trader: Paper trading engine
            fill_simulator: Fill simulation for paper orders
            api_client: Live API client
            is_paper: Whether to use paper trading
        """
        self.paper_trader = paper_trader
        self.fill_simulator = fill_simulator or FillSimulator()
        self.api_client = api_client
        self.is_paper = is_paper
    
    async def execute(self, order: Order, orderbook=None) -> OrderResult:
        """
        Execute order against exchange or paper.
        
        Args:
            order: Order to execute
            orderbook: Current orderbook (for paper/routing)
            
        Returns:
            OrderResult with execution details
        """
        if self.is_paper:
            return await self.execute_paper(order, orderbook)
        else:
            return await self.execute_live(order)
    
    async def execute_paper(self, order: Order, orderbook=None) -> OrderResult:
        """
        Execute against paper trading system.
        
        Args:
            order: Order to execute
            orderbook: Orderbook for fill simulation
        """
        try:
            order.submit()
            
            if order.order_type == OrderType.MARKET:
                return await self._execute_paper_market(order, orderbook)
            else:
                return await self._execute_paper_limit(order, orderbook)
            
        except Exception as e:
            logger.error(f"Paper execution failed: {e}")
            order.reject(str(e))
            return OrderResult(
                success=False,
                order=order,
                message="Paper execution failed",
                error=str(e)
            )
    
    async def _execute_paper_market(self, order: Order, orderbook) -> OrderResult:
        """Execute paper market order."""
        if orderbook is None:
            # Simulate at estimated price
            price = 0.50
            fill = Fill(
                price=price,
                size=order.size,
                timestamp=datetime.utcnow(),
                fee=order.size * price * 0.02
            )
            order.add_fill(fill)
            
            return OrderResult(
                success=True,
                order=order,
                message=f"Paper market order filled at {price:.4f}"
            )
        
        # Use fill simulator
        result = self.fill_simulator.simulate_fill(
            orderbook=orderbook,
            side="buy" if order.action == "BUY" else "sell",
            size=order.size
        )
        
        if result.filled_size == 0:
            order.reject("No liquidity available")
            return OrderResult(
                success=False,
                order=order,
                message="Market order failed - no liquidity",
                error="No liquidity"
            )
        
        fill = Fill(
            price=result.average_price,
            size=result.filled_size,
            timestamp=datetime.utcnow(),
            fee=result.fees
        )
        order.add_fill(fill)
        
        if result.unfilled_size > 0:
            return OrderResult(
                success=True,
                order=order,
                message=f"Partial fill: {result.filled_size}/{order.size} at {result.average_price:.4f}"
            )
        
        return OrderResult(
            success=True,
            order=order,
            message=f"Filled at {result.average_price:.4f}"
        )
    
    async def _execute_paper_limit(self, order: Order, orderbook) -> OrderResult:
        """Execute paper limit order."""
        if orderbook is None:
            # Check if limit would fill at assumed price
            mid_price = 0.50
            
            can_fill = (
                (order.action == "BUY" and order.limit_price >= mid_price) or
                (order.action == "SELL" and order.limit_price <= mid_price)
            )
            
            if can_fill:
                fill = Fill(
                    price=order.limit_price,
                    size=order.size,
                    timestamp=datetime.utcnow(),
                    fee=order.size * order.limit_price * 0.02
                )
                order.add_fill(fill)
                
                return OrderResult(
                    success=True,
                    order=order,
                    message=f"Limit order filled at {order.limit_price:.4f}"
                )
            else:
                # Order rests on book
                return OrderResult(
                    success=True,
                    order=order,
                    message=f"Limit order submitted at {order.limit_price:.4f}"
                )
        
        # Use fill simulator with limit
        result = self.fill_simulator.simulate_fill(
            orderbook=orderbook,
            side="buy" if order.action == "BUY" else "sell",
            size=order.size,
            limit_price=order.limit_price
        )
        
        if result.filled_size > 0:
            fill = Fill(
                price=result.average_price,
                size=result.filled_size,
                timestamp=datetime.utcnow(),
                fee=result.fees
            )
            order.add_fill(fill)
        
        if result.filled_size == 0:
            return OrderResult(
                success=True,
                order=order,
                message=f"Limit order resting at {order.limit_price:.4f}"
            )
        elif result.unfilled_size > 0:
            return OrderResult(
                success=True,
                order=order,
                message=f"Partial fill: {result.filled_size}/{order.size}"
            )
        else:
            return OrderResult(
                success=True,
                order=order,
                message=f"Filled at {result.average_price:.4f}"
            )
    
    async def execute_live(self, order: Order) -> OrderResult:
        """
        Execute against Polymarket via LiveTradingClient.
        
        Args:
            order: Order to execute
        """
        if self.api_client is None:
            order.reject("No API client configured")
            return OrderResult(
                success=False,
                order=order,
                message="Live execution not configured",
                error="No API client"
            )
        
        try:
            order.submit()
            
            # Get token_id from order (market_id should map to token)
            token_id = order.token_id if hasattr(order, 'token_id') else order.market_id
            
            # Determine action side for Polymarket
            # In Polymarket: BUY YES = buy shares, SELL YES = sell shares
            side = order.action  # "BUY" or "SELL"
            
            # Submit to exchange via LiveTradingClient
            if order.order_type == OrderType.MARKET:
                result = self.api_client.create_market_order(
                    token_id=token_id,
                    side=side,
                    amount=order.size * (order.limit_price or 0.50),  # Dollar amount
                )
            else:
                result = self.api_client.create_limit_order(
                    token_id=token_id,
                    side=side,
                    size=order.size,
                    price=order.limit_price,
                )
            
            if result.success:
                # Parse fill information
                if result.fills:
                    for f in result.fills:
                        fill = Fill(
                            price=float(f.get("price", order.limit_price or 0.50)),
                            size=float(f.get("size", order.size)),
                            timestamp=datetime.utcnow(),
                            fee=float(f.get("fee", 0))
                        )
                        order.add_fill(fill)
                else:
                    # If no fill info but success, assume full fill at limit
                    fill = Fill(
                        price=order.limit_price or 0.50,
                        size=order.size,
                        timestamp=datetime.utcnow(),
                        fee=order.size * (order.limit_price or 0.50) * 0.02
                    )
                    order.add_fill(fill)
                
                logger.info(f"Live order executed: {result.order_id}")
                return OrderResult(
                    success=True,
                    order=order,
                    message=f"Order executed: {result.message}"
                )
            else:
                order.reject(result.error or "Unknown error")
                return OrderResult(
                    success=False,
                    order=order,
                    message="Order rejected by exchange",
                    error=result.error
                )
                
        except Exception as e:
            logger.error(f"Live execution failed: {e}")
            order.reject(str(e))
            return OrderResult(
                success=False,
                order=order,
                message="Live execution failed",
                error=str(e)
            )
    
    def _build_order_payload(self, order: Order) -> dict:
        """Build API payload for order."""
        payload = {
            "market": order.market_id,
            "side": order.side.lower(),
            "type": order.action.lower(),
            "size": str(order.size),
        }
        
        if order.order_type == OrderType.LIMIT:
            payload["price"] = str(order.limit_price)
        
        if order.expires_at:
            payload["expiration"] = order.expires_at.isoformat()
        
        return payload
    
    async def cancel_order(self, order: Order) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order: Order to cancel
            
        Returns:
            True if cancelled successfully
        """
        if self.is_paper:
            order.cancel("Cancelled by user")
            return True
        
        if self.api_client is None:
            return False
        
        try:
            response = await self.api_client.cancel_order(order.id)
            if response.get("success"):
                order.cancel("Cancelled by user")
                return True
            return False
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

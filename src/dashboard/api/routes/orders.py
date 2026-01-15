"""
Orders API routes.
"""
from typing import List
from fastapi import APIRouter, HTTPException, Query

from src.dashboard.api.models import (
    OrderResponse,
    TradeResponse,
    CreateOrderRequest,
)
from src.dashboard.api.dependencies import get_order_manager
from src.execution.orders import OrderType

router = APIRouter(prefix="/api/orders", tags=["Orders"])


@router.get("", response_model=List[OrderResponse])
async def get_orders(
    status: str = Query("all", description="Filter: pending, active, filled, cancelled, all"),
    limit: int = Query(100, ge=1, le=500, description="Max results")
) -> List[OrderResponse]:
    """
    Get orders with filtering.
    
    - **status**: pending, active, filled, cancelled, all
    - **limit**: Maximum results
    """
    manager = get_order_manager()
    
    if status == "pending":
        orders = manager.tracker.get_pending_orders()
    elif status == "active":
        orders = manager.tracker.get_active_orders()
    elif status == "filled" or status == "cancelled":
        orders = [
            o for o in manager.tracker.completed_orders
            if o.status.value == status
        ]
    else:
        orders = list(manager.tracker.active_orders.values()) + \
                 manager.tracker.completed_orders
    
    return [
        OrderResponse(
            id=o.id,
            market_id=o.market_id,
            side=o.side,
            action=o.action,
            order_type=o.order_type.value,
            size=o.size,
            filled_size=o.filled_size,
            limit_price=o.limit_price,
            average_fill_price=o.average_fill_price,
            status=o.status.value,
            created_at=o.created_at
        )
        for o in orders[:limit]
    ]


@router.post("", response_model=OrderResponse)
async def create_order(request: CreateOrderRequest) -> OrderResponse:
    """
    Create new order.
    
    - **market_id**: Market to trade
    - **side**: YES or NO
    - **action**: BUY or SELL
    - **order_type**: market or limit
    - **size**: Order size
    - **limit_price**: Price for limit orders
    """
    manager = get_order_manager()
    
    order = manager.create_order(
        market_id=request.market_id,
        side=request.side,
        action=request.action,
        size=request.size,
        order_type=OrderType(request.order_type),
        limit_price=request.limit_price
    )
    
    result = await manager.submit_order(order)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return OrderResponse(
        id=order.id,
        market_id=order.market_id,
        side=order.side,
        action=order.action,
        order_type=order.order_type.value,
        size=order.size,
        filled_size=order.filled_size,
        limit_price=order.limit_price,
        average_fill_price=order.average_fill_price,
        status=order.status.value,
        created_at=order.created_at
    )


@router.delete("/{order_id}", response_model=OrderResponse)
async def cancel_order(order_id: str) -> OrderResponse:
    """
    Cancel order.
    
    - **order_id**: Order to cancel
    """
    manager = get_order_manager()
    
    order = manager.tracker.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    success = await manager.cancel_order(order_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel order")
    
    return OrderResponse(
        id=order.id,
        market_id=order.market_id,
        side=order.side,
        action=order.action,
        order_type=order.order_type.value,
        size=order.size,
        filled_size=order.filled_size,
        limit_price=order.limit_price,
        average_fill_price=order.average_fill_price,
        status=order.status.value,
        created_at=order.created_at
    )

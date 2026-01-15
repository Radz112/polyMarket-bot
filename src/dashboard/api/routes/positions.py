"""
Positions API routes.
"""
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Query, Depends

from src.dashboard.api.models import (
    PositionResponse,
    PositionDetailResponse,
    TradeResponse,
    ReducePositionRequest,
    PricePoint,
)
from src.dashboard.api.dependencies import get_position_manager

router = APIRouter(prefix="/api/positions", tags=["Positions"])


@router.get("", response_model=List[PositionResponse])
async def get_positions(
    status: str = Query("open", description="Filter: open, closed, all")
) -> List[PositionResponse]:
    """
    Get positions with filtering.
    
    - **status**: open, closed, or all
    """
    manager = get_position_manager()
    
    if status == "open":
        positions = manager.get_all_positions()
    elif status == "closed":
        positions = [cp.position for cp in manager.closed_positions]
    else:
        positions = manager.get_all_positions() + \
                   [cp.position for cp in manager.closed_positions]
    
    return [
        PositionResponse(
            id=p.id,
            market_id=p.market_id,
            market_name=p.market_name,
            side=p.side,
            size=p.size,
            entry_price=p.entry_price,
            current_price=p.current_price,
            unrealized_pnl=p.unrealized_pnl,
            unrealized_pnl_pct=p.unrealized_pnl_pct,
            opened_at=p.entry_time
        )
        for p in positions
    ]


@router.get("/{position_id}", response_model=PositionDetailResponse)
async def get_position(position_id: str) -> PositionDetailResponse:
    """
    Get position details with history.
    
    - **position_id**: Position identifier
    """
    manager = get_position_manager()
    position = manager.get_position_by_id(position_id)
    
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    return PositionDetailResponse(
        position=PositionResponse(
            id=position.id,
            market_id=position.market_id,
            market_name=position.market_name,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            current_price=position.current_price,
            unrealized_pnl=position.unrealized_pnl,
            unrealized_pnl_pct=position.unrealized_pnl_pct,
            opened_at=position.entry_time
        ),
        price_history=[],
        max_gain=0,
        max_loss=0,
        alerts=[]
    )


@router.post("/{position_id}/close", response_model=TradeResponse)
async def close_position(position_id: str) -> TradeResponse:
    """
    Close position by submitting an offsetting order.
    """
    from src.dashboard.api.dependencies import get_order_manager
    from src.execution.orders.models import OrderType
    
    pos_manager = get_position_manager()
    order_manager = get_order_manager()
    
    position = pos_manager.get_position_by_id(position_id)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    try:
        # Determine order parameters to close the position
        # If LONG YES, we need to SELL YES
        # If SHORT NO, we need to SELL NO (if supported)
        
        order = order_manager.create_order(
            market_id=position.market_id,
            side=position.side,
            action="SELL",
            size=position.size,
            order_type=OrderType.MARKET,
            signal_id=position.signal_id
        )
        
        result = await order_manager.submit_order(order)
        
        if result.success:
            # Note: PositionManager will be updated via OrderManager fill tracking or manual close
            # For now, we also manually update the local position state for paper-like responsiveness
            await pos_manager.close_position(position_id, position.current_price, "manual_close")
            
            return TradeResponse(
                success=True,
                position_id=position_id,
                order_id=order.id,
                message=f"Closing order submitted: {result.message}"
            )
        else:
            return TradeResponse(
                success=False,
                message="Failed to submit closing order",
                error=result.error
            )
            
    except Exception as e:
        return TradeResponse(
            success=False,
            message="Error closing position",
            error=str(e)
        )


@router.post("/{position_id}/reduce", response_model=TradeResponse)
async def reduce_position(
    position_id: str,
    request: ReducePositionRequest
) -> TradeResponse:
    """
    Reduce position size by submitting an offsetting order.
    """
    from src.dashboard.api.dependencies import get_order_manager
    from src.execution.orders.models import OrderType
    
    pos_manager = get_position_manager()
    order_manager = get_order_manager()
    
    position = pos_manager.get_position_by_id(position_id)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    try:
        order = order_manager.create_order(
            market_id=position.market_id,
            side=position.side,
            action="SELL",
            size=request.size,
            order_type=OrderType.MARKET,
            signal_id=position.signal_id
        )
        
        result = await order_manager.submit_order(order)
        
        if result.success:
            # Manually update local state for UI responsiveness
            await pos_manager.reduce_position(position_id, request.size, position.current_price)
            
            return TradeResponse(
                success=True,
                position_id=position_id,
                order_id=order.id,
                message=f"Reduction order submitted: {result.message}"
            )
        else:
            return TradeResponse(
                success=False,
                message="Failed to submit reduction order",
                error=result.error
            )
            
    except Exception as e:
        return TradeResponse(
            success=False,
            message="Error reducing position",
            error=str(e)
        )

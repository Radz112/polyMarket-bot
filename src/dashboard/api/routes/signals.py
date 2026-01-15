"""
Signals API routes.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends

from src.dashboard.api.models import (
    SignalResponse,
    SignalDetailResponse,
    TradeResponse,
    TradeSignalRequest,
    DismissSignalRequest,
)
from src.dashboard.api.dependencies import get_order_manager

router = APIRouter(prefix="/api/signals", tags=["Signals"])


# In-memory mock signals
_mock_signals = {}
_dismissed_signals = set()


@router.get("", response_model=List[SignalResponse])
async def get_signals(
    status: str = Query("active", description="Filter by status"),
    min_score: float = Query(0, ge=0, le=100, description="Minimum score"),
    limit: int = Query(50, ge=1, le=200, description="Max results")
) -> List[SignalResponse]:
    """
    Get signals with filtering.
    
    - **status**: active, traded, expired, all
    - **min_score**: Minimum signal score
    - **limit**: Maximum results
    """
    # Would fetch from signal store
    signals = list(_mock_signals.values())
    
    if status != "all":
        signals = [s for s in signals if s.get("status") == status]
    
    signals = [s for s in signals if s.get("score", 0) >= min_score]
    
    return [SignalResponse(**s) for s in signals[:limit]]


@router.get("/{signal_id}", response_model=SignalDetailResponse)
async def get_signal(signal_id: str) -> SignalDetailResponse:
    """
    Get signal details with scoring breakdown.
    
    - **signal_id**: Signal identifier
    """
    signal = _mock_signals.get(signal_id)
    
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    
    return SignalDetailResponse(
        signal=SignalResponse(**signal),
        component_scores={},
        risk_factors=[]
    )


@router.post("/{signal_id}/trade", response_model=TradeResponse)
async def trade_signal(
    signal_id: str,
    request: TradeSignalRequest = None
) -> TradeResponse:
    """
    Execute trade on signal.
    
    - **signal_id**: Signal to trade
    - **size**: Optional size override
    """
    # 1. Get signal (currently from mocks, in prod from signal store)
    signal_data = _mock_signals.get(signal_id)
    
    if not signal_data:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    
    if signal_id in _dismissed_signals:
        return TradeResponse(
            success=False,
            message="Signal was dismissed",
            error="Signal already dismissed"
        )
    
    # 2. Prepare trade parameters
    size_override = request.size if request else None
    
    # 3. Execute via OrderManager
    from src.execution.orders.models import OrderType
    manager = get_order_manager()
    
    try:
        # Create a mock internal signal object for submit_signal_order
        # or use create_order directly
        order = manager.create_order(
            market_id=signal_data["markets"][0]["id"], # Use primary market
            side="YES", # Default for mock signals
            action=signal_data["recommendedAction"],
            size=size_override or signal_data["recommendedSize"],
            order_type=OrderType.LIMIT if signal_data.get("recommendedPrice") else OrderType.MARKET,
            limit_price=signal_data.get("recommendedPrice"),
            signal_id=signal_id
        )
        
        result = await manager.submit_order(order)
        
        if result.success:
            # Update status in mock storage
            signal_data["status"] = "traded"
            return TradeResponse(
                success=True,
                order_id=order.id,
                message=f"Trade executed: {result.message}"
            )
        else:
            return TradeResponse(
                success=False,
                message="Trade execution failed",
                error=result.error
            )
            
    except Exception as e:
        return TradeResponse(
            success=False,
            message="Error submitting trade",
            error=str(e)
        )


@router.post("/{signal_id}/dismiss")
async def dismiss_signal(
    signal_id: str,
    request: DismissSignalRequest = None
) -> dict:
    """
    Dismiss signal without trading.
    
    - **signal_id**: Signal to dismiss
    - **reason**: Optional dismissal reason
    """
    if signal_id in _mock_signals:
        _dismissed_signals.add(signal_id)
    
    return {"message": f"Signal {signal_id} dismissed"}

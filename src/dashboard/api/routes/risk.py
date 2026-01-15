"""
Risk API routes.
"""
from typing import List
from fastapi import APIRouter

from src.dashboard.api.models import (
    RiskDashboardResponse,
    BreakerStatusResponse,
    HaltTradingRequest,
)
from src.dashboard.api.dependencies import app_state, get_position_manager, get_config

router = APIRouter(prefix="/api/risk", tags=["Risk"])


@router.get("", response_model=RiskDashboardResponse)
async def get_risk_dashboard() -> RiskDashboardResponse:
    """
    Get current risk state.
    
    Returns exposure, P&L, drawdown, and breaker status.
    """
    config = get_config()
    manager = get_position_manager()
    
    total_exposure = manager.get_total_exposure()
    initial = config.paper_trading_balance
    exposure_pct = (total_exposure / initial) * 100 if initial > 0 else 0
    
    # Calculate daily P&L (would need daily tracking)
    daily_pnl = sum(p.unrealized_pnl for p in manager.get_all_positions())
    daily_pnl_pct = (daily_pnl / initial) * 100 if initial > 0 else 0
    
    # Risk score (0-100)
    risk_score = min(100, int(exposure_pct + abs(daily_pnl_pct) * 2))
    
    if risk_score < 30:
        risk_level = "low"
    elif risk_score < 60:
        risk_level = "medium"
    elif risk_score < 80:
        risk_level = "high"
    else:
        risk_level = "critical"
    
    return RiskDashboardResponse(
        trading_allowed=not app_state.trading_halted,
        halt_reasons=app_state.halt_reasons,
        total_exposure=total_exposure,
        exposure_pct=exposure_pct,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=0,  # Would calculate from history
        risk_score=risk_score,
        risk_level=risk_level,
        breakers={}
    )


@router.get("/breakers", response_model=List[BreakerStatusResponse])
async def get_circuit_breakers() -> List[BreakerStatusResponse]:
    """
    Get circuit breaker states.
    """
    # Would fetch from risk manager
    return [
        BreakerStatusResponse(name="daily_loss", status="armed"),
        BreakerStatusResponse(name="max_drawdown", status="armed"),
        BreakerStatusResponse(name="position_limit", status="armed"),
    ]


@router.post("/breakers/{breaker_name}/reset")
async def reset_breaker(breaker_name: str) -> dict:
    """
    Reset circuit breaker.
    
    - **breaker_name**: Breaker to reset
    """
    return {"message": f"Breaker {breaker_name} reset"}


@router.post("/halt")
async def halt_trading(request: HaltTradingRequest) -> dict:
    """
    Emergency halt trading.
    
    - **reason**: Reason for halt
    """
    app_state.trading_halted = True
    app_state.halt_reasons.append(request.reason)
    
    return {"message": "Trading halted", "reason": request.reason}


@router.post("/resume")
async def resume_trading() -> dict:
    """
    Resume trading after halt.
    """
    app_state.trading_halted = False
    app_state.halt_reasons.clear()
    
    return {"message": "Trading resumed"}

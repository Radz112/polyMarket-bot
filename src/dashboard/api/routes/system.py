"""
System API routes.
"""
import time
from fastapi import APIRouter

from src.dashboard.api.models import HealthResponse, StatsResponse
from src.dashboard.api.dependencies import app_state, get_position_manager

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System health check.
    
    Returns status, version, uptime, and service health.
    """
    uptime = time.time() - app_state.start_time if app_state.start_time else 0
    
    # Check services
    services = {
        "api": "healthy",
        "database": "unknown",  # Would check DB
        "redis": "unknown",     # Would check Redis
    }
    
    # Determine overall status
    if all(s == "healthy" for s in services.values()):
        status = "healthy"
    elif any(s == "unhealthy" for s in services.values()):
        status = "unhealthy"
    else:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=uptime,
        services=services
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    System statistics.
    
    Returns counts for markets, correlations, signals, trades, positions, orders.
    """
    manager = get_position_manager()
    
    return StatsResponse(
        markets_tracked=0,  # Would fetch from market store
        correlations_found=0,  # Would fetch from correlation store
        signals_today=0,  # Would fetch from signal store
        trades_today=len(manager.closed_positions),
        active_positions=len(manager.positions),
        pending_orders=0  # Would fetch from order manager
    )

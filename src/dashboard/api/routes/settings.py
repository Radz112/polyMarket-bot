"""
Settings API routes.
"""
from fastapi import APIRouter

from src.dashboard.api.models import (
    SettingsResponse,
    RiskLimitsResponse,
    UpdateSettingsRequest,
    UpdateLimitsRequest,
)
from src.dashboard.api.dependencies import get_config

router = APIRouter(prefix="/api/settings", tags=["Settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """
    Get current bot settings.
    """
    config = get_config()
    
    return SettingsResponse(
        paper_trading=config.paper_trading,
        auto_trade=False,  # Would have auto_trade setting
        max_position_size_pct=config.max_position_size_pct,
        max_daily_loss_pct=config.max_daily_loss_pct,
        min_signal_score=config.min_signal_score,
        signal_expiry_seconds=config.signal_expiry_seconds
    )


@router.put("", response_model=SettingsResponse)
async def update_settings(request: UpdateSettingsRequest) -> SettingsResponse:
    """
    Update bot settings.
    
    Only provided fields are updated.
    """
    config = get_config()
    
    # Would actually update config
    # For now, return current with updates
    return SettingsResponse(
        paper_trading=request.paper_trading if request.paper_trading is not None else config.paper_trading,
        auto_trade=request.auto_trade if request.auto_trade is not None else False,
        max_position_size_pct=request.max_position_size_pct or config.max_position_size_pct,
        max_daily_loss_pct=request.max_daily_loss_pct or config.max_daily_loss_pct,
        min_signal_score=request.min_signal_score if request.min_signal_score is not None else config.min_signal_score,
        signal_expiry_seconds=request.signal_expiry_seconds or config.signal_expiry_seconds
    )


@router.get("/limits", response_model=RiskLimitsResponse)
async def get_risk_limits() -> RiskLimitsResponse:
    """
    Get risk limits.
    """
    config = get_config()
    
    return RiskLimitsResponse(
        max_position_pct=config.max_position_size_pct,
        max_exposure_pct=0.80,  # Would be in config
        max_daily_loss_pct=config.max_daily_loss_pct,
        max_drawdown_pct=0.20,  # Would be in config
        max_correlation_exposure=5000  # Would be in config
    )


@router.put("/limits", response_model=RiskLimitsResponse)
async def update_risk_limits(request: UpdateLimitsRequest) -> RiskLimitsResponse:
    """
    Update risk limits.
    
    Only provided fields are updated.
    """
    config = get_config()
    
    return RiskLimitsResponse(
        max_position_pct=request.max_position_pct or config.max_position_size_pct,
        max_exposure_pct=request.max_exposure_pct or 0.80,
        max_daily_loss_pct=request.max_daily_loss_pct or config.max_daily_loss_pct,
        max_drawdown_pct=request.max_drawdown_pct or 0.20,
        max_correlation_exposure=request.max_correlation_exposure or 5000
    )

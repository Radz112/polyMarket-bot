"""
Request models for Dashboard API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class CreateOrderRequest(BaseModel):
    """Request to create a new order."""
    market_id: str
    side: str = Field(..., pattern="^(YES|NO)$")
    action: str = Field(..., pattern="^(BUY|SELL)$")
    order_type: str = Field(default="limit", pattern="^(market|limit)$")
    size: float = Field(..., gt=0)
    limit_price: Optional[float] = Field(None, gt=0, lt=1)


class TradeSignalRequest(BaseModel):
    """Request to trade a signal."""
    size: Optional[float] = Field(None, gt=0)


class ReducePositionRequest(BaseModel):
    """Request to reduce position."""
    size: float = Field(..., gt=0)


class DismissSignalRequest(BaseModel):
    """Request to dismiss a signal."""
    reason: Optional[str] = None


class HaltTradingRequest(BaseModel):
    """Request to halt trading."""
    reason: str


class VerifyCorrelationRequest(BaseModel):
    """Request to verify correlation."""
    verified: bool


class UpdateSettingsRequest(BaseModel):
    """Request to update settings."""
    paper_trading: Optional[bool] = None
    auto_trade: Optional[bool] = None
    max_position_size_pct: Optional[float] = Field(None, gt=0, le=1)
    max_daily_loss_pct: Optional[float] = Field(None, gt=0, le=1)
    min_signal_score: Optional[float] = Field(None, ge=0, le=100)
    signal_expiry_seconds: Optional[int] = Field(None, gt=0)


class UpdateLimitsRequest(BaseModel):
    """Request to update risk limits."""
    max_position_pct: Optional[float] = Field(None, gt=0, le=1)
    max_exposure_pct: Optional[float] = Field(None, gt=0, le=1)
    max_daily_loss_pct: Optional[float] = Field(None, gt=0, le=1)
    max_drawdown_pct: Optional[float] = Field(None, gt=0, le=1)
    max_correlation_exposure: Optional[float] = Field(None, gt=0)

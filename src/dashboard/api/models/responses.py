"""
Response models for Dashboard API.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


# ============ Market Models ============

class MarketResponse(BaseModel):
    """Market summary response."""
    id: str
    slug: str
    question: str
    category: Optional[str] = ""
    end_date: Optional[datetime] = None
    active: bool
    yes_price: float
    no_price: float
    volume_24h: float = 0

    class Config:
        from_attributes = True


class CorrelationResponse(BaseModel):
    """Correlation between markets."""
    id: str
    market_a_id: str
    market_b_id: str
    market_a_name: str = ""
    market_b_name: str = ""
    correlation_type: str
    confidence: float
    expected_relationship: str = ""
    verified: bool = False
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MarketDetailResponse(BaseModel):
    """Detailed market response with correlations."""
    market: MarketResponse
    correlations: List[CorrelationResponse] = []
    orderbook: Optional[Dict[str, Any]] = None


class PricePoint(BaseModel):
    """Price history point."""
    timestamp: datetime
    price: float
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    volume: float = 0


# ============ Signal Models ============

class SignalMarket(BaseModel):
    """Market info within a signal."""
    id: str
    question: str
    yes_price: float
    no_price: float


class SignalResponse(BaseModel):
    """Signal summary response."""
    id: str
    signal_type: str
    markets: List[SignalMarket]
    divergence_amount: float
    score: float
    recommended_action: str
    recommended_size: float
    recommended_price: Optional[float] = None
    detected_at: datetime
    expires_at: Optional[datetime] = None
    status: str = "active"
    
    class Config:
        from_attributes = True


class SignalDetailResponse(BaseModel):
    """Detailed signal with scoring breakdown."""
    signal: SignalResponse
    component_scores: Dict[str, float] = {}
    risk_factors: List[str] = []


# ============ Position Models ============

class PositionResponse(BaseModel):
    """Position summary response."""
    id: str
    market_id: str
    market_name: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: datetime
    
    class Config:
        from_attributes = True


class PositionDetailResponse(BaseModel):
    """Detailed position with history."""
    position: PositionResponse
    price_history: List[PricePoint] = []
    max_gain: float = 0
    max_loss: float = 0
    alerts: List[str] = []


# ============ Order Models ============

class OrderResponse(BaseModel):
    """Order summary response."""
    id: str
    market_id: str
    side: str
    action: str
    order_type: str
    size: float
    filled_size: float
    limit_price: Optional[float] = None
    average_fill_price: Optional[float] = None
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Portfolio Models ============

class PortfolioResponse(BaseModel):
    """Current portfolio state."""
    cash_balance: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    return_pct: float
    positions: List[PositionResponse] = []


class PortfolioSnapshot(BaseModel):
    """Portfolio snapshot for history."""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float


class PerformanceResponse(BaseModel):
    """Performance metrics."""
    total_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    total_return: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float] = None


# ============ Risk Models ============

class BreakerStatusResponse(BaseModel):
    """Circuit breaker status."""
    name: str
    status: str  # "armed", "tripped", "disabled"
    tripped_at: Optional[datetime] = None
    reason: Optional[str] = None


class RiskDashboardResponse(BaseModel):
    """Risk dashboard overview."""
    trading_allowed: bool
    halt_reasons: List[str] = []
    total_exposure: float
    exposure_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    drawdown_pct: float
    risk_score: int  # 0-100
    risk_level: str  # "low", "medium", "high", "critical"
    breakers: Dict[str, str] = {}


# ============ Settings Models ============

class SettingsResponse(BaseModel):
    """Bot settings."""
    paper_trading: bool
    auto_trade: bool
    max_position_size_pct: float
    max_daily_loss_pct: float
    min_signal_score: float
    signal_expiry_seconds: int


class RiskLimitsResponse(BaseModel):
    """Risk limits."""
    max_position_pct: float
    max_exposure_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_correlation_exposure: float


# ============ System Models ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime_seconds: float
    services: Dict[str, str] = {}


class StatsResponse(BaseModel):
    """System statistics."""
    markets_tracked: int
    correlations_found: int
    signals_today: int
    trades_today: int
    active_positions: int
    pending_orders: int


# ============ Trade Response ============

class TradeResponse(BaseModel):
    """Response for trade execution."""
    success: bool
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    message: str
    error: Optional[str] = None

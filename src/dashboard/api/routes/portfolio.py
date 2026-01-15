"""
Portfolio API routes.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Query

from src.dashboard.api.models import (
    PortfolioResponse,
    PortfolioSnapshot,
    PerformanceResponse,
    PositionResponse,
)
from src.dashboard.api.dependencies import get_position_manager, get_config

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("", response_model=PortfolioResponse)
async def get_portfolio() -> PortfolioResponse:
    """
    Get current portfolio state.
    
    Returns cash balance, positions value, P&L, and all positions.
    """
    config = get_config()
    manager = get_position_manager()
    
    positions = manager.get_all_positions()
    positions_value = sum(p.market_value for p in positions)
    unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    
    # For paper trading, use initial balance
    initial_balance = config.paper_trading_balance
    cash_balance = initial_balance - sum(p.cost_basis for p in positions)
    total_value = cash_balance + positions_value
    
    # Realized P&L from closed positions
    realized_pnl = sum(cp.realized_pnl for cp in manager.closed_positions)
    
    total_pnl = realized_pnl + unrealized_pnl
    return_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0
    
    position_responses = [
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
    
    return PortfolioResponse(
        cash_balance=cash_balance,
        positions_value=positions_value,
        total_value=total_value,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        total_pnl=total_pnl,
        return_pct=return_pct,
        positions=position_responses
    )


@router.get("/history", response_model=List[PortfolioSnapshot])
async def get_portfolio_history(
    start: Optional[datetime] = Query(None, description="Start time"),
    end: Optional[datetime] = Query(None, description="End time"),
    interval: str = Query("1h", description="Interval: 1m, 5m, 1h, 1d")
) -> List[PortfolioSnapshot]:
    """
    Get portfolio value history for equity curve.
    
    - **start**: Start timestamp
    - **end**: End timestamp  
    - **interval**: Time interval
    """
    # Would fetch from portfolio tracker
    return []


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance() -> PerformanceResponse:
    """
    Get performance metrics.
    
    Returns win rate, average win/loss, Sharpe ratio, etc.
    """
    manager = get_position_manager()
    closed = manager.closed_positions
    
    if not closed:
        return PerformanceResponse(
            total_trades=0,
            win_rate=0,
            average_win=0,
            average_loss=0,
            profit_factor=0,
            total_return=0,
            total_return_pct=0,
            max_drawdown_pct=0,
            sharpe_ratio=None
        )
    
    wins = [cp for cp in closed if cp.realized_pnl > 0]
    losses = [cp for cp in closed if cp.realized_pnl < 0]
    
    win_rate = len(wins) / len(closed) if closed else 0
    avg_win = sum(cp.realized_pnl for cp in wins) / len(wins) if wins else 0
    avg_loss = sum(cp.realized_pnl for cp in losses) / len(losses) if losses else 0
    
    total_wins = sum(cp.realized_pnl for cp in wins)
    total_losses = abs(sum(cp.realized_pnl for cp in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    total_return = sum(cp.realized_pnl for cp in closed)
    config = get_config()
    total_return_pct = (total_return / config.paper_trading_balance) * 100
    
    return PerformanceResponse(
        total_trades=len(closed),
        win_rate=win_rate,
        average_win=avg_win,
        average_loss=avg_loss,
        profit_factor=profit_factor,
        total_return=total_return,
        total_return_pct=total_return_pct,
        max_drawdown_pct=0,  # Would calculate from history
        sharpe_ratio=None
    )

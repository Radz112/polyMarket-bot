"""
API routes module.
"""
from .markets import router as markets_router
from .signals import router as signals_router
from .positions import router as positions_router
from .portfolio import router as portfolio_router
from .orders import router as orders_router
from .risk import router as risk_router
from .settings import router as settings_router
from .system import router as system_router

__all__ = [
    "markets_router",
    "signals_router",
    "positions_router",
    "portfolio_router",
    "orders_router",
    "risk_router",
    "settings_router",
    "system_router",
]

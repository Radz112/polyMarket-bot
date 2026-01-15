"""
Paper trading module.

Provides simulated trading capabilities for testing strategies without real money.
"""
from .types import (
    PositionStatus,
    FillResult,
    PaperPosition,
    PaperTrade,
    PnLSummary,
    PortfolioState,
    PortfolioSnapshot,
    PerformanceMetrics,
)
from .fill_simulator import FillSimulator, SlippageModel
from .trader import PaperTrader
from .portfolio import PortfolioTracker
from .resolution import ResolutionHandler, monitor_pending_resolutions
from .persistence import PaperTradingPersistence

__all__ = [
    # Types
    "PositionStatus",
    "FillResult",
    "PaperPosition",
    "PaperTrade",
    "PnLSummary",
    "PortfolioState",
    "PortfolioSnapshot",
    "PerformanceMetrics",
    # Fill simulation
    "FillSimulator",
    "SlippageModel",
    # Trading
    "PaperTrader",
    # Portfolio
    "PortfolioTracker",
    # Resolution
    "ResolutionHandler",
    "monitor_pending_resolutions",
    # Persistence
    "PaperTradingPersistence",
]

"""
Divergence detection module.

Detects when correlated markets diverge from their expected relationships,
identifying potential arbitrage and trading opportunities.
"""
from .types import (
    DivergenceType,
    DivergenceStatus,
    Divergence,
    DivergenceConfig,
)
from .detector import DivergenceDetector
from .price_monitor import PriceMonitor
from .liquidity import (
    LiquidityAssessor,
    LiquidityAnalysis,
    TwoSidedLiquidity,
)

__all__ = [
    # Types
    "DivergenceType",
    "DivergenceStatus",
    "Divergence",
    "DivergenceConfig",
    # Detector
    "DivergenceDetector",
    # Price monitoring
    "PriceMonitor",
    # Liquidity
    "LiquidityAssessor",
    "LiquidityAnalysis",
    "TwoSidedLiquidity",
]

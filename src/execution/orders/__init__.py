"""
Order management module.

Provides order execution with validation, smart routing, and tracking.
"""
from .models import (
    OrderType,
    OrderStatus,
    Order,
    Fill,
    OrderResult,
    ValidationResult,
    ExecutionPlan,
)
from .validator import OrderValidator
from .executor import OrderExecutor
from .router import SmartOrderRouter
from .tracker import OrderTracker
from .manager import OrderManager

__all__ = [
    # Models
    "OrderType",
    "OrderStatus",
    "Order",
    "Fill",
    "OrderResult",
    "ValidationResult",
    "ExecutionPlan",
    # Components
    "OrderValidator",
    "OrderExecutor",
    "SmartOrderRouter",
    "OrderTracker",
    "OrderManager",
]

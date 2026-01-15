"""
Signal monitoring module.

Provides real-time signal detection, scoring, and lifecycle management.
"""
from src.signals.monitor.signal_monitor import SignalMonitor, MonitorConfig
from src.signals.monitor.lifecycle import (
    SignalState,
    SignalLifecycle,
    LifecycleManager,
    StateTransition,
)
from src.signals.monitor.deduplicator import (
    SignalDeduplicator,
    DeduplicationConfig,
    MarketSetDeduplicator,
)
from src.signals.monitor.optimizer import (
    MonitorOptimizer,
    OptimizerConfig,
    CorrelationBatcher,
    MarketPriority,
)
from src.signals.monitor.metrics import (
    MonitorMetrics,
    PerformanceTimer,
    MemoryTracker,
    Counter,
    Gauge,
    Histogram,
)
from src.signals.monitor.event_handler import (
    PriceEventHandler,
    EventHandlerConfig,
    PriceUpdate,
    TradeEvent,
    PriceSpike,
    EventAggregator,
)

__all__ = [
    # Main monitor
    "SignalMonitor",
    "MonitorConfig",
    # Lifecycle
    "SignalState",
    "SignalLifecycle",
    "LifecycleManager",
    "StateTransition",
    # Deduplication
    "SignalDeduplicator",
    "DeduplicationConfig",
    "MarketSetDeduplicator",
    # Optimization
    "MonitorOptimizer",
    "OptimizerConfig",
    "CorrelationBatcher",
    "MarketPriority",
    # Metrics
    "MonitorMetrics",
    "PerformanceTimer",
    "MemoryTracker",
    "Counter",
    "Gauge",
    "Histogram",
    # Event handling
    "PriceEventHandler",
    "EventHandlerConfig",
    "PriceUpdate",
    "TradeEvent",
    "PriceSpike",
    "EventAggregator",
]

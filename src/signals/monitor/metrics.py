"""
Monitor metrics and observability.

Provides comprehensive metrics collection for monitoring performance.
"""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

from src.signals.scoring.types import ScoredSignal, RecommendedAction
from src.signals.monitor.lifecycle import SignalState

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Counter:
    """Simple counter metric."""
    name: str
    value: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def inc(self, amount: int = 1) -> None:
        self.value += amount

    def reset(self) -> None:
        self.value = 0


@dataclass
class Gauge:
    """Gauge metric (current value)."""
    name: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class Histogram:
    """Histogram metric for distribution tracking."""
    name: str
    buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    values: List[float] = field(default_factory=list)
    max_values: int = 10000

    def observe(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.max_values:
            self.values = self.values[-self.max_values:]

    def get_percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def get_stats(self) -> Dict[str, float]:
        if not self.values:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
        return {
            "count": len(self.values),
            "mean": self.get_mean(),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "max": max(self.values),
        }


class MonitorMetrics:
    """
    Comprehensive metrics collection for the signal monitor.

    Tracks:
    - Signal detection counts and rates
    - Detection latency
    - Signal lifecycle metrics
    - Performance metrics
    """

    def __init__(self):
        # Counters
        self.signals_detected = Counter("signals_detected")
        self.signals_by_type: Dict[str, Counter] = defaultdict(
            lambda: Counter("signals_by_type")
        )
        self.signals_by_action: Dict[str, Counter] = defaultdict(
            lambda: Counter("signals_by_action")
        )
        self.signals_expired = Counter("signals_expired")
        self.signals_converged = Counter("signals_converged")
        self.signals_traded = Counter("signals_traded")
        self.duplicates_detected = Counter("duplicates_detected")
        self.detection_cycles = Counter("detection_cycles")
        self.errors = Counter("errors")

        # Gauges
        self.active_signal_count = Gauge("active_signal_count")
        self.correlations_monitored = Gauge("correlations_monitored")
        self.hot_markets_count = Gauge("hot_markets_count")
        self.memory_usage_mb = Gauge("memory_usage_mb")

        # Histograms
        self.signals_by_score = Histogram(
            "signals_by_score",
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        )
        self.detection_latency_ms = Histogram(
            "detection_latency_ms",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]
        )
        self.cycle_duration_ms = Histogram(
            "cycle_duration_ms",
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        )
        self.signal_age_seconds = Histogram(
            "signal_age_seconds",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
        )

        # Rate tracking
        self._detection_times: List[datetime] = []
        self._rate_window = timedelta(minutes=5)

        # Session tracking
        self.session_start = datetime.utcnow()
        self.last_detection_time: Optional[datetime] = None

    def record_detection(self, signal: ScoredSignal) -> None:
        """Record a signal detection."""
        now = datetime.utcnow()

        # Counters
        self.signals_detected.inc()

        # By type
        dtype = signal.divergence.divergence_type.value
        self.signals_by_type[dtype].inc()

        # By action
        action = signal.recommended_action.value
        self.signals_by_action[action].inc()

        # Score histogram
        self.signals_by_score.observe(signal.overall_score)

        # Track detection time for rate calculation
        self._detection_times.append(now)
        self._cleanup_rate_times()

        self.last_detection_time = now

        logger.debug(
            f"Recorded detection: {dtype} score={signal.overall_score:.1f} "
            f"action={action}"
        )

    def record_detection_cycle(
        self,
        duration_ms: float,
        signals_found: int,
        correlations_checked: int
    ) -> None:
        """Record metrics for a detection cycle."""
        self.detection_cycles.inc()
        self.cycle_duration_ms.observe(duration_ms)

        # Calculate per-correlation latency
        if correlations_checked > 0:
            per_corr_latency = duration_ms / correlations_checked
            self.detection_latency_ms.observe(per_corr_latency)

        logger.debug(
            f"Detection cycle: {duration_ms:.1f}ms, "
            f"{signals_found} signals, {correlations_checked} correlations"
        )

    def record_signal_closed(
        self,
        signal: ScoredSignal,
        state: SignalState,
        age_seconds: float
    ) -> None:
        """Record a signal being closed."""
        self.signal_age_seconds.observe(age_seconds)

        if state == SignalState.EXPIRED:
            self.signals_expired.inc()
        elif state == SignalState.CONVERGED:
            self.signals_converged.inc()
        elif state == SignalState.TRADED:
            self.signals_traded.inc()

    def record_duplicate(self) -> None:
        """Record a duplicate signal detection."""
        self.duplicates_detected.inc()

    def record_error(self, error_type: str = "unknown") -> None:
        """Record an error."""
        self.errors.inc()
        logger.warning(f"Error recorded: {error_type}")

    def update_gauges(
        self,
        active_signals: int,
        correlations: int = 0,
        hot_markets: int = 0
    ) -> None:
        """Update current value gauges."""
        self.active_signal_count.set(active_signals)
        if correlations > 0:
            self.correlations_monitored.set(correlations)
        if hot_markets > 0:
            self.hot_markets_count.set(hot_markets)

        # Track memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage_mb.set(memory_mb)
        except ImportError:
            pass

    def get_detection_rate(self) -> float:
        """Get signals detected per minute."""
        self._cleanup_rate_times()
        if not self._detection_times:
            return 0.0

        window_seconds = self._rate_window.total_seconds()
        return len(self._detection_times) / (window_seconds / 60)

    def _cleanup_rate_times(self) -> None:
        """Remove old detection times."""
        cutoff = datetime.utcnow() - self._rate_window
        self._detection_times = [t for t in self._detection_times if t > cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        uptime = (datetime.utcnow() - self.session_start).total_seconds()

        return {
            "session": {
                "start_time": self.session_start.isoformat(),
                "uptime_seconds": uptime,
                "last_detection": self.last_detection_time.isoformat() if self.last_detection_time else None,
            },
            "counters": {
                "signals_detected": self.signals_detected.value,
                "detection_cycles": self.detection_cycles.value,
                "signals_expired": self.signals_expired.value,
                "signals_converged": self.signals_converged.value,
                "signals_traded": self.signals_traded.value,
                "duplicates_detected": self.duplicates_detected.value,
                "errors": self.errors.value,
            },
            "gauges": {
                "active_signal_count": self.active_signal_count.value,
                "correlations_monitored": self.correlations_monitored.value,
                "hot_markets_count": self.hot_markets_count.value,
                "memory_usage_mb": self.memory_usage_mb.value,
            },
            "rates": {
                "detections_per_minute": self.get_detection_rate(),
                "cycles_per_minute": self.detection_cycles.value / (uptime / 60) if uptime > 0 else 0,
            },
            "by_type": {
                dtype: counter.value
                for dtype, counter in self.signals_by_type.items()
            },
            "by_action": {
                action: counter.value
                for action, counter in self.signals_by_action.items()
            },
            "latency": self.detection_latency_ms.get_stats(),
            "cycle_duration": self.cycle_duration_ms.get_stats(),
            "signal_age": self.signal_age_seconds.get_stats(),
            "score_distribution": self.signals_by_score.get_stats(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()

        lines = [
            f"Monitor Metrics Summary",
            f"=" * 40,
            f"Uptime: {stats['session']['uptime_seconds']:.0f}s",
            f"",
            f"Signals:",
            f"  Detected: {stats['counters']['signals_detected']}",
            f"  Active: {stats['gauges']['active_signal_count']:.0f}",
            f"  Rate: {stats['rates']['detections_per_minute']:.2f}/min",
            f"",
            f"Outcomes:",
            f"  Converged: {stats['counters']['signals_converged']}",
            f"  Expired: {stats['counters']['signals_expired']}",
            f"  Traded: {stats['counters']['signals_traded']}",
            f"",
            f"Performance:",
            f"  Cycles: {stats['counters']['detection_cycles']}",
            f"  Latency (p95): {stats['latency']['p95']:.1f}ms",
            f"  Cycle duration (p95): {stats['cycle_duration']['p95']:.1f}ms",
            f"  Memory: {stats['gauges']['memory_usage_mb']:.1f}MB",
        ]

        if stats['counters']['errors'] > 0:
            lines.append(f"  Errors: {stats['counters']['errors']}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        self.signals_detected.reset()
        self.signals_expired.reset()
        self.signals_converged.reset()
        self.signals_traded.reset()
        self.duplicates_detected.reset()
        self.detection_cycles.reset()
        self.errors.reset()

        self.signals_by_type.clear()
        self.signals_by_action.clear()

        self.signals_by_score.values.clear()
        self.detection_latency_ms.values.clear()
        self.cycle_duration_ms.values.clear()
        self.signal_age_seconds.values.clear()

        self._detection_times.clear()
        self.session_start = datetime.utcnow()
        self.last_detection_time = None


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000


class MemoryTracker:
    """Tracks memory usage over time for leak detection."""

    def __init__(self, sample_interval_seconds: int = 60):
        self.sample_interval = sample_interval_seconds
        self.samples: List[tuple[datetime, float]] = []
        self.max_samples = 1000
        self.last_sample: Optional[datetime] = None

    def sample(self) -> Optional[float]:
        """Take a memory sample if interval has passed."""
        now = datetime.utcnow()

        if self.last_sample:
            elapsed = (now - self.last_sample).total_seconds()
            if elapsed < self.sample_interval:
                return None

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            self.samples.append((now, memory_mb))
            if len(self.samples) > self.max_samples:
                self.samples = self.samples[-self.max_samples:]

            self.last_sample = now
            return memory_mb
        except ImportError:
            return None

    def detect_leak(self, threshold_mb_per_hour: float = 10.0) -> bool:
        """
        Detect potential memory leaks.

        Returns True if memory growth exceeds threshold.
        """
        if len(self.samples) < 10:
            return False

        # Check growth over last hour
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent = [(t, m) for t, m in self.samples if t > cutoff]

        if len(recent) < 5:
            return False

        # Calculate growth rate
        first_mem = recent[0][1]
        last_mem = recent[-1][1]
        time_span_hours = (recent[-1][0] - recent[0][0]).total_seconds() / 3600

        if time_span_hours < 0.1:  # Less than 6 minutes
            return False

        growth_rate = (last_mem - first_mem) / time_span_hours
        return growth_rate > threshold_mb_per_hour

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.samples:
            return {"samples": 0}

        memories = [m for _, m in self.samples]
        return {
            "samples": len(self.samples),
            "current_mb": memories[-1],
            "min_mb": min(memories),
            "max_mb": max(memories),
            "avg_mb": sum(memories) / len(memories),
        }

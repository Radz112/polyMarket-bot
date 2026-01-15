"""
Main signal monitoring loop.

Continuously detects, scores, and tracks divergence signals.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set

from src.signals.divergence.detector import DivergenceDetector
from src.signals.divergence.types import Divergence, DivergenceStatus
from src.signals.scoring.scorer import SignalScorer
from src.signals.scoring.types import ScoredSignal, RecommendedAction
from src.correlation.store import CorrelationStore
from src.signals.monitor.lifecycle import (
    SignalState,
    SignalLifecycle,
    LifecycleManager,
)
from src.signals.monitor.deduplicator import SignalDeduplicator, DeduplicationConfig
from src.signals.monitor.optimizer import MonitorOptimizer, OptimizerConfig
from src.signals.monitor.metrics import MonitorMetrics, PerformanceTimer

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for the signal monitor."""
    # Detection intervals
    detection_interval_seconds: float = 2.0  # Main detection loop interval
    cleanup_interval_seconds: float = 60.0    # Cleanup loop interval

    # Signal filtering
    min_signal_score: float = 40.0  # Minimum score to track
    min_actionable_score: float = 60.0  # Minimum score to recommend action

    # Limits
    max_active_signals: int = 100  # Maximum concurrent signals
    max_signals_per_cycle: int = 20  # Max new signals per detection cycle

    # Performance
    enable_batching: bool = True
    batch_size: int = 50
    parallel_checks: bool = True

    # Logging
    log_detections: bool = True
    log_interval_seconds: int = 60  # Log stats every N seconds


class SignalMonitor:
    """
    Main monitoring loop for divergence detection and scoring.

    Features:
    - Continuous detection of divergences
    - Real-time scoring and ranking
    - Signal lifecycle management
    - Event-driven callbacks
    - Performance optimization
    """

    def __init__(
        self,
        divergence_detector: DivergenceDetector,
        signal_scorer: SignalScorer,
        correlation_store: CorrelationStore,
        cache=None,  # CacheManager
        config: MonitorConfig = None,
    ):
        self.detector = divergence_detector
        self.scorer = signal_scorer
        self.correlation_store = correlation_store
        self.cache = cache
        self.config = config or MonitorConfig()

        # State
        self.running = False
        self.active_signals: Dict[str, ScoredSignal] = {}

        # Components
        self.lifecycle_manager = LifecycleManager()
        self.deduplicator = SignalDeduplicator()
        self.optimizer = MonitorOptimizer()
        self.metrics = MonitorMetrics()

        # Callbacks
        self._on_signal_callbacks: List[Callable[[ScoredSignal], Any]] = []
        self._on_update_callbacks: List[Callable[[ScoredSignal], Any]] = []
        self._on_expired_callbacks: List[Callable[[ScoredSignal], Any]] = []
        self._on_converged_callbacks: List[Callable[[ScoredSignal], Any]] = []

        # Tasks
        self._detection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

        # Tracking
        self._last_stats_log = datetime.utcnow()
        self._correlations_loaded = False

    async def start(self) -> None:
        """
        Start the monitoring loop.

        1. Load all correlations and rules
        2. Build optimization index
        3. Start detection loop
        4. Start cleanup loop
        """
        if self.running:
            logger.warning("Monitor is already running")
            return

        logger.info("Starting signal monitor...")

        # Load correlations
        await self._load_correlations()

        # Start loops
        self.running = True

        self._detection_task = asyncio.create_task(
            self._detection_loop(),
            name="detection_loop"
        )

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="cleanup_loop"
        )

        if self.config.log_interval_seconds > 0:
            self._stats_task = asyncio.create_task(
                self._stats_loop(),
                name="stats_loop"
            )

        logger.info("Signal monitor started")

    async def stop(self) -> None:
        """Gracefully stop monitoring."""
        if not self.running:
            return

        logger.info("Stopping signal monitor...")
        self.running = False

        # Cancel tasks
        for task in [self._detection_task, self._cleanup_task, self._stats_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(f"Signal monitor stopped. Final stats:\n{self.metrics.get_summary()}")

    async def _load_correlations(self) -> None:
        """Load correlations and build optimization index."""
        logger.info("Loading correlations...")

        correlations = await self.correlation_store.get_all_active()
        self.optimizer.build_market_index(correlations)

        self._correlations_loaded = True
        self.metrics.correlations_monitored.set(len(correlations))

        logger.info(f"Loaded {len(correlations)} correlations")

    async def _detection_loop(self) -> None:
        """
        Main detection loop.

        Runs every N seconds, detects divergences, scores them,
        and handles new signals.
        """
        logger.info(f"Detection loop started (interval: {self.config.detection_interval_seconds}s)")

        while self.running:
            try:
                with PerformanceTimer("detection_cycle") as timer:
                    await self._run_detection_cycle()

                self.metrics.record_detection_cycle(
                    duration_ms=timer.duration_ms,
                    signals_found=len(self.active_signals),
                    correlations_checked=self.optimizer._correlation_count,
                )

                # Update gauges
                self.metrics.update_gauges(
                    active_signals=len(self.active_signals),
                    correlations=self.optimizer._correlation_count,
                    hot_markets=len(self.optimizer.hot_markets),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in detection loop: {e}", exc_info=True)
                self.metrics.record_error("detection_loop")

            await asyncio.sleep(self.config.detection_interval_seconds)

    async def _run_detection_cycle(self) -> None:
        """Run a single detection cycle."""
        # Detect all divergences
        divergences = await self.detector.detect_all_divergences()

        new_signal_count = 0

        for div in divergences:
            # Check limits
            if len(self.active_signals) >= self.config.max_active_signals:
                logger.debug("Max active signals reached, skipping")
                break

            if new_signal_count >= self.config.max_signals_per_cycle:
                logger.debug("Max signals per cycle reached")
                break

            # Score the divergence
            signal = self.scorer.score_divergence(div)

            # Filter by minimum score
            if signal.overall_score < self.config.min_signal_score:
                continue

            # Handle the signal
            is_new = await self._handle_signal(signal)
            if is_new:
                new_signal_count += 1

        # Update existing signals
        await self._update_active_signals()

    async def _handle_signal(self, signal: ScoredSignal) -> bool:
        """
        Handle a detected signal.

        Returns True if signal is new, False if duplicate/update.
        """
        signal_id = signal.divergence.id

        # Check for duplicate
        is_dup, existing_id = self.deduplicator.is_duplicate(
            signal, self.active_signals
        )

        if is_dup and existing_id:
            # Update existing signal
            await self._update_existing_signal(existing_id, signal)
            self.metrics.record_duplicate()
            return False

        # New signal
        await self._add_new_signal(signal)
        return True

    async def _add_new_signal(self, signal: ScoredSignal) -> None:
        """Add a new signal to tracking."""
        signal_id = signal.divergence.id

        # Create lifecycle
        lifecycle = self.lifecycle_manager.add(signal)
        lifecycle.activate("Passed minimum score filter")

        # Register for deduplication
        self.deduplicator.register_signal(signal)

        # Add to active signals
        self.active_signals[signal_id] = signal

        # Record metrics
        self.metrics.record_detection(signal)

        # Log if enabled
        if self.config.log_detections:
            logger.info(
                f"New signal: {signal.divergence.divergence_type.value} "
                f"score={signal.overall_score:.1f} "
                f"action={signal.recommended_action.value} "
                f"profit={signal.divergence.profit_potential*100:.2f}¢"
            )

        # Emit callback
        await self._emit_signal_event(signal)

    async def _update_existing_signal(
        self,
        existing_id: str,
        new_signal: ScoredSignal
    ) -> None:
        """Update an existing signal with new data."""
        if existing_id not in self.active_signals:
            return

        existing = self.active_signals[existing_id]
        lifecycle = self.lifecycle_manager.get(existing_id)

        if not lifecycle:
            return

        # Check update cooldown
        if not self.deduplicator.should_update(existing_id, lifecycle.last_updated):
            return

        # Merge signals
        merged = self.deduplicator.merge_signals(existing, new_signal)

        # Update in tracking
        self.active_signals[existing_id] = merged
        lifecycle.update_signal(merged)

        # Check for significant score change
        score_change = abs(merged.overall_score - existing.overall_score)
        if score_change >= 5.0:  # Significant change
            await self._emit_update_event(merged)

    async def _update_active_signals(self) -> None:
        """
        Update all active signals with current prices.

        For each active signal:
        1. Get current prices
        2. Check if expired or converged
        3. Update if needed
        """
        signals_to_remove: List[str] = []

        for signal_id, signal in list(self.active_signals.items()):
            lifecycle = self.lifecycle_manager.get(signal_id)
            if not lifecycle:
                signals_to_remove.append(signal_id)
                continue

            # Check expiration
            if lifecycle.should_expire():
                await self._expire_signal(signal_id, lifecycle)
                signals_to_remove.append(signal_id)
                continue

            # Check convergence (simplified - would need current prices)
            current_div_pct = signal.divergence.divergence_pct
            if lifecycle.has_converged(current_div_pct):
                await self._converge_signal(signal_id, lifecycle, current_div_pct)
                signals_to_remove.append(signal_id)
                continue

        # Remove closed signals
        for signal_id in signals_to_remove:
            self._remove_signal(signal_id)

    async def _expire_signal(
        self,
        signal_id: str,
        lifecycle: SignalLifecycle
    ) -> None:
        """Expire a signal."""
        signal = self.active_signals.get(signal_id)
        if not signal:
            return

        lifecycle.mark_expired()

        self.metrics.record_signal_closed(
            signal,
            SignalState.EXPIRED,
            lifecycle.get_age_seconds()
        )

        logger.debug(f"Signal expired: {signal_id}")

        await self._emit_expired_event(signal)

    async def _converge_signal(
        self,
        signal_id: str,
        lifecycle: SignalLifecycle,
        final_divergence: float
    ) -> None:
        """Mark signal as converged."""
        signal = self.active_signals.get(signal_id)
        if not signal:
            return

        lifecycle.mark_converged(final_divergence)

        self.metrics.record_signal_closed(
            signal,
            SignalState.CONVERGED,
            lifecycle.get_age_seconds()
        )

        logger.info(f"Signal converged: {signal_id}")

        await self._emit_converged_event(signal)

    def _remove_signal(self, signal_id: str) -> None:
        """Remove a signal from tracking."""
        if signal_id in self.active_signals:
            signal = self.active_signals.pop(signal_id)
            self.deduplicator.unregister_signal(signal)

        self.lifecycle_manager.remove(signal_id)

    async def _cleanup_loop(self) -> None:
        """
        Cleanup loop.

        - Remove terminal signals
        - Clean deduplication cache
        - Archive old signals
        """
        logger.info(f"Cleanup loop started (interval: {self.config.cleanup_interval_seconds}s)")

        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                # Cleanup terminal lifecycles
                terminal = self.lifecycle_manager.cleanup_terminal()
                if terminal:
                    logger.debug(f"Cleaned up {len(terminal)} terminal signals")

                # Cleanup deduplication cache
                self.deduplicator.cleanup_old_entries()

                # TODO: Archive to database

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                self.metrics.record_error("cleanup_loop")

    async def _stats_loop(self) -> None:
        """Periodic stats logging."""
        while self.running:
            try:
                await asyncio.sleep(self.config.log_interval_seconds)

                logger.info(f"\n{self.metrics.get_summary()}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")

    # Event handling
    def on_signal(self, callback: Callable[[ScoredSignal], Any]) -> None:
        """Register callback for new signals."""
        self._on_signal_callbacks.append(callback)

    def on_signal_update(self, callback: Callable[[ScoredSignal], Any]) -> None:
        """Register callback for signal updates."""
        self._on_update_callbacks.append(callback)

    def on_signal_expired(self, callback: Callable[[ScoredSignal], Any]) -> None:
        """Register callback for expired signals."""
        self._on_expired_callbacks.append(callback)

    def on_signal_converged(self, callback: Callable[[ScoredSignal], Any]) -> None:
        """Register callback for converged signals."""
        self._on_converged_callbacks.append(callback)

    async def _emit_signal_event(self, signal: ScoredSignal) -> None:
        """Emit signal event to callbacks."""
        for callback in self._on_signal_callbacks:
            try:
                result = callback(signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    async def _emit_update_event(self, signal: ScoredSignal) -> None:
        """Emit update event to callbacks."""
        for callback in self._on_update_callbacks:
            try:
                result = callback(signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Update callback error: {e}")

    async def _emit_expired_event(self, signal: ScoredSignal) -> None:
        """Emit expired event to callbacks."""
        for callback in self._on_expired_callbacks:
            try:
                result = callback(signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Expired callback error: {e}")

    async def _emit_converged_event(self, signal: ScoredSignal) -> None:
        """Emit converged event to callbacks."""
        for callback in self._on_converged_callbacks:
            try:
                result = callback(signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Converged callback error: {e}")

    # Public API
    async def scan_correlated_markets(self, market_id: str) -> List[ScoredSignal]:
        """
        Scan correlations involving a specific market.

        Useful for event-driven updates when a market's price changes.
        """
        correlations = self.optimizer.get_correlations_for_market(market_id)
        if not correlations:
            return []

        # Check each correlation for divergence
        detected = []
        for corr in correlations:
            div = await self.detector.check_correlation(corr)
            if div:
                signal = self.scorer.score_divergence(div)
                if signal.overall_score >= self.config.min_signal_score:
                    detected.append(signal)
                    await self._handle_signal(signal)

        return detected

    async def update_signals_for_markets(self, market_ids: List[str]) -> None:
        """
        Update signals involving specific markets.

        Called by event handler when prices change.
        """
        for signal_id, signal in list(self.active_signals.items()):
            signal_markets = set(signal.divergence.market_ids)
            if signal_markets & set(market_ids):
                # Re-score this signal
                new_signal = self.scorer.score_divergence(signal.divergence)
                await self._update_existing_signal(signal_id, new_signal)

    def get_active_signals(
        self,
        min_score: float = None,
        action_filter: RecommendedAction = None,
    ) -> List[ScoredSignal]:
        """Get active signals, optionally filtered."""
        signals = list(self.active_signals.values())

        if min_score is not None:
            signals = [s for s in signals if s.overall_score >= min_score]

        if action_filter is not None:
            signals = [s for s in signals if s.recommended_action == action_filter]

        # Sort by score descending
        return sorted(signals, key=lambda s: s.overall_score, reverse=True)

    def get_actionable_signals(self) -> List[ScoredSignal]:
        """Get signals above actionable threshold."""
        return self.get_active_signals(min_score=self.config.min_actionable_score)

    def get_signal(self, signal_id: str) -> Optional[ScoredSignal]:
        """Get a specific signal by ID."""
        return self.active_signals.get(signal_id)

    def get_lifecycle(self, signal_id: str) -> Optional[SignalLifecycle]:
        """Get lifecycle for a signal."""
        return self.lifecycle_manager.get(signal_id)

    def mark_signal_traded(
        self,
        signal_id: str,
        trade_id: str = None,
        trade_size: float = 0.0
    ) -> bool:
        """Mark a signal as traded."""
        lifecycle = self.lifecycle_manager.get(signal_id)
        if not lifecycle:
            return False

        return lifecycle.mark_traded(trade_id, trade_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitor statistics."""
        return {
            "running": self.running,
            "active_signals": len(self.active_signals),
            "lifecycle_counts": self.lifecycle_manager.count_by_state(),
            "optimizer": self.optimizer.get_stats(),
            "deduplicator": self.deduplicator.get_stats(),
            "metrics": self.metrics.get_stats(),
        }

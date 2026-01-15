"""
Event-driven price update handling.

Handles real-time price events from WebSocket feeds.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Callable, Any

from src.models import Orderbook

if TYPE_CHECKING:
    from src.signals.monitor.signal_monitor import SignalMonitor

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
    """Represents a price update event."""
    market_id: str
    timestamp: datetime
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class TradeEvent:
    """Represents a trade event."""
    market_id: str
    timestamp: datetime
    price: float
    size: float
    side: str  # "buy" or "sell"


@dataclass
class PriceSpike:
    """Represents a significant price move."""
    market_id: str
    timestamp: datetime
    old_price: float
    new_price: float
    change_pct: float
    direction: str  # "up" or "down"


@dataclass
class EventHandlerConfig:
    """Configuration for the price event handler."""
    # Price spike detection
    spike_threshold_pct: float = 0.02  # 2% move = spike
    spike_window_seconds: int = 30  # Within this time window

    # Large trade detection
    large_trade_threshold: float = 500.0  # $500+ = large

    # Debouncing
    min_update_interval_ms: int = 100  # Max 10 updates/second per market

    # Batch processing
    batch_window_ms: int = 50  # Batch updates within this window


class PriceEventHandler:
    """
    Handles real-time price events and triggers signal evaluation.

    Features:
    - Orderbook update processing
    - Trade event handling
    - Price spike detection
    - Debouncing for high-frequency updates
    """

    def __init__(
        self,
        signal_monitor: "SignalMonitor",
        config: EventHandlerConfig = None
    ):
        self.monitor = signal_monitor
        self.config = config or EventHandlerConfig()

        # Price tracking for spike detection
        self._recent_prices: Dict[str, List[tuple[datetime, float]]] = {}

        # Debouncing
        self._last_update: Dict[str, datetime] = {}
        self._pending_updates: Dict[str, PriceUpdate] = {}

        # Callbacks
        self._on_spike_callbacks: List[Callable] = []
        self._on_large_trade_callbacks: List[Callable] = []

        # Batch processing
        self._update_batch: List[str] = []
        self._batch_task: Optional[asyncio.Task] = None

        # Statistics
        self._updates_processed = 0
        self._updates_debounced = 0
        self._spikes_detected = 0

    async def on_orderbook_update(
        self,
        market_id: str,
        orderbook: Orderbook
    ) -> None:
        """
        Called when orderbook updates via WebSocket.

        1. Update cache
        2. Check for price spike
        3. Check if any active signals involve this market
        4. Trigger re-evaluation if needed
        """
        now = datetime.utcnow()

        # Debounce check
        if not self._should_process_update(market_id, now):
            self._updates_debounced += 1
            return

        self._updates_processed += 1
        self._last_update[market_id] = now

        # Extract price info
        mid_price = None
        if orderbook.bids and orderbook.asks:
            best_bid = float(orderbook.bids[0].price)
            best_ask = float(orderbook.asks[0].price)
            mid_price = (best_bid + best_ask) / 2

        # Check for price spike
        if mid_price is not None:
            spike = await self._check_price_spike(market_id, mid_price, now)
            if spike:
                await self._handle_price_spike(spike)

        # Update cache in monitor
        if hasattr(self.monitor, 'cache') and self.monitor.cache:
            await self.monitor.cache.set(
                f"orderbook:{market_id}",
                orderbook,
                ttl=30
            )

        # Check if any active signals involve this market
        if self._market_has_active_signals(market_id):
            # Add to batch for processing
            self._add_to_batch(market_id)

    async def on_trade(
        self,
        market_id: str,
        price: float,
        size: float,
        side: str = "unknown"
    ) -> None:
        """
        Called when a trade occurs.

        Large trades might indicate:
        - News event
        - Whale activity
        - Potential lagging market opportunity
        """
        now = datetime.utcnow()

        trade = TradeEvent(
            market_id=market_id,
            timestamp=now,
            price=price,
            size=size,
            side=side,
        )

        # Check if large trade
        if size >= self.config.large_trade_threshold:
            await self._handle_large_trade(trade)

        # Update recent price
        self._record_price(market_id, price, now)

    async def on_price_spike(
        self,
        market_id: str,
        old_price: float,
        new_price: float
    ) -> None:
        """
        Called when price moves significantly (>threshold) in short time.

        Trigger immediate scan for lagging correlated markets.
        """
        change_pct = abs(new_price - old_price) / old_price if old_price > 0 else 0
        direction = "up" if new_price > old_price else "down"

        spike = PriceSpike(
            market_id=market_id,
            timestamp=datetime.utcnow(),
            old_price=old_price,
            new_price=new_price,
            change_pct=change_pct,
            direction=direction,
        )

        await self._handle_price_spike(spike)

    def register_spike_callback(
        self,
        callback: Callable[[PriceSpike], Any]
    ) -> None:
        """Register callback for price spikes."""
        self._on_spike_callbacks.append(callback)

    def register_large_trade_callback(
        self,
        callback: Callable[[TradeEvent], Any]
    ) -> None:
        """Register callback for large trades."""
        self._on_large_trade_callbacks.append(callback)

    def _should_process_update(
        self,
        market_id: str,
        now: datetime
    ) -> bool:
        """Check if update should be processed (debouncing)."""
        if market_id not in self._last_update:
            return True

        elapsed_ms = (now - self._last_update[market_id]).total_seconds() * 1000
        return elapsed_ms >= self.config.min_update_interval_ms

    def _record_price(
        self,
        market_id: str,
        price: float,
        timestamp: datetime
    ) -> None:
        """Record price for spike detection."""
        if market_id not in self._recent_prices:
            self._recent_prices[market_id] = []

        self._recent_prices[market_id].append((timestamp, price))

        # Keep only recent prices (within spike window)
        cutoff = timestamp - timedelta(seconds=self.config.spike_window_seconds * 2)
        self._recent_prices[market_id] = [
            (t, p) for t, p in self._recent_prices[market_id]
            if t > cutoff
        ]

    async def _check_price_spike(
        self,
        market_id: str,
        new_price: float,
        now: datetime
    ) -> Optional[PriceSpike]:
        """Check if current price represents a spike."""
        if market_id not in self._recent_prices:
            self._record_price(market_id, new_price, now)
            return None

        recent = self._recent_prices[market_id]
        if not recent:
            self._record_price(market_id, new_price, now)
            return None

        # Find oldest price within window
        cutoff = now - timedelta(seconds=self.config.spike_window_seconds)
        window_prices = [(t, p) for t, p in recent if t > cutoff]

        if not window_prices:
            self._record_price(market_id, new_price, now)
            return None

        # Compare to oldest price in window
        old_time, old_price = window_prices[0]

        if old_price <= 0:
            self._record_price(market_id, new_price, now)
            return None

        change_pct = abs(new_price - old_price) / old_price

        # Record new price
        self._record_price(market_id, new_price, now)

        if change_pct >= self.config.spike_threshold_pct:
            self._spikes_detected += 1
            return PriceSpike(
                market_id=market_id,
                timestamp=now,
                old_price=old_price,
                new_price=new_price,
                change_pct=change_pct,
                direction="up" if new_price > old_price else "down",
            )

        return None

    async def _handle_price_spike(self, spike: PriceSpike) -> None:
        """Handle a detected price spike."""
        logger.info(
            f"Price spike detected: {spike.market_id} "
            f"{spike.direction} {spike.change_pct:.1%} "
            f"({spike.old_price:.4f} -> {spike.new_price:.4f})"
        )

        # Call registered callbacks
        for callback in self._on_spike_callbacks:
            try:
                result = callback(spike)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Spike callback error: {e}")

        # Trigger scan for lagging markets
        if hasattr(self.monitor, 'scan_correlated_markets'):
            await self.monitor.scan_correlated_markets(spike.market_id)

    async def _handle_large_trade(self, trade: TradeEvent) -> None:
        """Handle a large trade event."""
        logger.info(
            f"Large trade: {trade.market_id} "
            f"${trade.size:.0f} @ {trade.price:.4f} ({trade.side})"
        )

        # Call registered callbacks
        for callback in self._on_large_trade_callbacks:
            try:
                result = callback(trade)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Large trade callback error: {e}")

    def _market_has_active_signals(self, market_id: str) -> bool:
        """Check if market is involved in any active signals."""
        if not hasattr(self.monitor, 'active_signals'):
            return False

        for signal in self.monitor.active_signals.values():
            if market_id in signal.divergence.market_ids:
                return True

        return False

    def _add_to_batch(self, market_id: str) -> None:
        """Add market to update batch."""
        if market_id not in self._update_batch:
            self._update_batch.append(market_id)

        # Start batch processing if not already running
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._process_batch())

    async def _process_batch(self) -> None:
        """Process batched updates."""
        # Wait for batch window
        await asyncio.sleep(self.config.batch_window_ms / 1000)

        if not self._update_batch:
            return

        # Get and clear batch
        markets_to_update = self._update_batch.copy()
        self._update_batch.clear()

        # Trigger signal updates for affected markets
        if hasattr(self.monitor, 'update_signals_for_markets'):
            await self.monitor.update_signals_for_markets(markets_to_update)

    def get_stats(self) -> Dict[str, Any]:
        """Get event handler statistics."""
        return {
            "updates_processed": self._updates_processed,
            "updates_debounced": self._updates_debounced,
            "spikes_detected": self._spikes_detected,
            "markets_tracked": len(self._recent_prices),
            "pending_batch_size": len(self._update_batch),
        }


class EventAggregator:
    """
    Aggregates multiple events for batch processing.

    Useful for reducing load when many markets update simultaneously.
    """

    def __init__(
        self,
        window_ms: int = 100,
        on_batch: Optional[Callable] = None
    ):
        self.window_ms = window_ms
        self.on_batch = on_batch

        self._events: Dict[str, Any] = {}  # market_id -> latest event
        self._timer_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def add_event(self, market_id: str, event: Any) -> None:
        """Add an event to the batch."""
        async with self._lock:
            self._events[market_id] = event

            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        """Flush events after delay."""
        await asyncio.sleep(self.window_ms / 1000)
        await self.flush()

    async def flush(self) -> None:
        """Process all pending events."""
        async with self._lock:
            if not self._events:
                return

            events = self._events.copy()
            self._events.clear()

        if self.on_batch:
            try:
                result = self.on_batch(events)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

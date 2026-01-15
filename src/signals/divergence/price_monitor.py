"""
Real-time price monitoring for divergence detection.

Integrates with Polymarket WebSocket to track live prices and orderbooks.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from collections import deque
from dataclasses import dataclass, field

from src.api.ws_client import ClobWsClient
from src.database.redis_cache import CacheManager
from src.models import Orderbook, OrderbookEntry, PriceSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: float  # Unix timestamp
    price: float
    orderbook: Optional[Orderbook] = None


class PriceMonitor:
    """
    Real-time price monitor for market divergence detection.

    Subscribes to market updates via WebSocket and maintains
    current prices and recent price history for all tracked markets.
    """

    def __init__(
        self,
        ws_client: ClobWsClient,
        cache: Optional[CacheManager] = None,
        history_window_seconds: int = 300,  # 5 minutes of history
        max_history_points: int = 1000,     # Per market
    ):
        self.ws = ws_client
        self.cache = cache
        self.history_window_seconds = history_window_seconds
        self.max_history_points = max_history_points

        # Current state - keyed by market_id (condition_id)
        self._market_prices: Dict[str, float] = {}
        self._market_orderbooks: Dict[str, Orderbook] = {}

        # Price history: market_id -> deque of PricePoints
        self._price_history: Dict[str, deque] = {}

        # Subscribed markets (market_ids/condition_ids)
        self._subscribed_markets: Set[str] = set()

        # Token ID mappings: token_id -> market_id (for reverse lookup)
        self._token_to_market: Dict[str, str] = {}
        # Market to token IDs: market_id -> [yes_token_id, no_token_id]
        self._market_to_tokens: Dict[str, List[str]] = {}

        # Callbacks for price updates
        self._on_update_callbacks: List[Callable[[str, float, Optional[Orderbook]], Any]] = []

        # Running state
        self._is_running = False
        self._update_lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Start the price monitor.

        Registers a handler with the WebSocket client to receive market updates.
        """
        if self._is_running:
            logger.warning("PriceMonitor already running")
            return

        self._is_running = True

        # Register our handler with the WebSocket client
        self.ws.register_callback(self._handle_ws_message)

        logger.info("PriceMonitor started")

    async def stop(self) -> None:
        """Stop the price monitor."""
        self._is_running = False
        logger.info("PriceMonitor stopped")

    async def subscribe_to_markets(self, market_ids: List[str]) -> None:
        """
        Subscribe to orderbook updates for specific markets (legacy method).
        Use subscribe_with_tokens for proper token ID handling.

        Args:
            market_ids: List of market/token IDs to subscribe to
        """
        new_markets = [mid for mid in market_ids if mid not in self._subscribed_markets]

        if not new_markets:
            return

        # Subscribe via WebSocket
        await self.ws.subscribe(new_markets)

        # Track subscriptions
        self._subscribed_markets.update(new_markets)

        # Initialize history for new markets
        for mid in new_markets:
            if mid not in self._price_history:
                self._price_history[mid] = deque(maxlen=self.max_history_points)

        logger.info(f"Subscribed to {len(new_markets)} new markets")

    async def subscribe_with_tokens(self, market_token_map: Dict[str, List[str]]) -> None:
        """
        Subscribe to markets using their CLOB token IDs.

        Args:
            market_token_map: Dict mapping market_id (condition_id) to list of token_ids
                             e.g., {"0xabc...": ["12345...", "67890..."]}
        """
        all_token_ids = []

        for market_id, token_ids in market_token_map.items():
            if market_id in self._subscribed_markets:
                continue

            # Store mappings
            self._market_to_tokens[market_id] = token_ids
            for token_id in token_ids:
                self._token_to_market[token_id] = market_id
                all_token_ids.append(token_id)

            # Track subscription
            self._subscribed_markets.add(market_id)

            # Initialize history
            if market_id not in self._price_history:
                self._price_history[market_id] = deque(maxlen=self.max_history_points)

        if all_token_ids:
            # Subscribe to all token IDs via WebSocket
            await self.ws.subscribe(all_token_ids)
            logger.info(f"Subscribed to {len(all_token_ids)} token IDs for {len(market_token_map)} markets")

    async def unsubscribe_from_markets(self, market_ids: List[str]) -> None:
        """Unsubscribe from market updates."""
        markets_to_remove = [mid for mid in market_ids if mid in self._subscribed_markets]

        if not markets_to_remove:
            return

        await self.ws.unsubscribe(markets_to_remove)
        self._subscribed_markets.difference_update(markets_to_remove)

        logger.info(f"Unsubscribed from {len(markets_to_remove)} markets")

    def on_price_update(self, callback: Callable[[str, float, Optional[Orderbook]], Any]) -> None:
        """
        Register a callback for price updates.

        The callback receives (market_id, price, orderbook).
        """
        self._on_update_callbacks.append(callback)

    async def _handle_ws_message(self, data) -> None:
        """
        Handle incoming WebSocket messages.

        Polymarket WS messages for market updates typically contain:
        - asset_id or market: the market identifier
        - Orderbook data with bids/asks
        """
        try:
            # Handle list of messages (batch updates)
            if isinstance(data, list):
                for item in data:
                    await self._handle_single_message(item)
            else:
                await self._handle_single_message(data)

        except Exception as e:
            logger.error(f"Error handling WS message: {e}")

    async def _handle_single_message(self, data: Dict[str, Any]) -> None:
        """Handle a single message dict."""
        if not isinstance(data, dict):
            return

        # Parse the message format
        # Polymarket uses different message types
        event_type = data.get("event_type") or data.get("type")

        if event_type == "book":
            await self._handle_orderbook_update(data)
        elif event_type == "price_change":
            await self._handle_price_change(data)
        elif event_type == "last_trade_price":
            await self._handle_last_trade(data)
        # Handle raw orderbook snapshot
        elif "bids" in data or "asks" in data:
            await self._handle_orderbook_update(data)

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Process an orderbook update message."""
        token_id = data.get("asset_id") or data.get("market") or data.get("token_id")
        if not token_id:
            return

        # Map token_id to market_id (condition_id) if we have a mapping
        market_id = self._token_to_market.get(token_id, token_id)

        # Parse orderbook
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Convert to OrderbookEntry format
        bid_entries = []
        ask_entries = []

        for bid in bids:
            if isinstance(bid, dict):
                bid_entries.append(OrderbookEntry(
                    price=float(bid.get("price", 0)),
                    size=float(bid.get("size", 0))
                ))
            elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                bid_entries.append(OrderbookEntry(
                    price=float(bid[0]),
                    size=float(bid[1])
                ))

        for ask in asks:
            if isinstance(ask, dict):
                ask_entries.append(OrderbookEntry(
                    price=float(ask.get("price", 0)),
                    size=float(ask.get("size", 0))
                ))
            elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                ask_entries.append(OrderbookEntry(
                    price=float(ask[0]),
                    size=float(ask[1])
                ))

        # Sort: bids descending, asks ascending
        bid_entries.sort(key=lambda x: x.price, reverse=True)
        ask_entries.sort(key=lambda x: x.price)

        orderbook = Orderbook(
            market_id=market_id,
            token_id=token_id,
            bids=bid_entries,
            asks=ask_entries
        )

        # Calculate mid price
        mid_price = orderbook.mid_price
        if mid_price is not None:
            await self._update_price(market_id, mid_price, orderbook)

    async def _handle_price_change(self, data: Dict[str, Any]) -> None:
        """Process a price change message."""
        token_id = data.get("asset_id") or data.get("market")
        market_id = self._token_to_market.get(token_id, token_id) if token_id else None
        price = data.get("price")

        if market_id and price is not None:
            await self._update_price(market_id, float(price), None)

    async def _handle_last_trade(self, data: Dict[str, Any]) -> None:
        """Process a last trade price message."""
        token_id = data.get("asset_id") or data.get("market")
        market_id = self._token_to_market.get(token_id, token_id) if token_id else None
        price = data.get("price")

        if market_id and price is not None:
            # Only update if we don't have orderbook-derived price
            if market_id not in self._market_prices:
                await self._update_price(market_id, float(price), None)

    async def _update_price(
        self,
        market_id: str,
        price: float,
        orderbook: Optional[Orderbook]
    ) -> None:
        """Update price and notify callbacks."""
        async with self._update_lock:
            now = time.time()

            # Update current state
            self._market_prices[market_id] = price
            if orderbook:
                self._market_orderbooks[market_id] = orderbook

            # Add to history
            if market_id not in self._price_history:
                self._price_history[market_id] = deque(maxlen=self.max_history_points)

            self._price_history[market_id].append(PricePoint(
                timestamp=now,
                price=price,
                orderbook=orderbook
            ))

            # Prune old history
            cutoff = now - self.history_window_seconds
            while (self._price_history[market_id] and
                   self._price_history[market_id][0].timestamp < cutoff):
                self._price_history[market_id].popleft()

            # Update cache if available
            if self.cache and orderbook:
                try:
                    await self.cache.set_orderbook(market_id, orderbook)
                except Exception as e:
                    logger.debug(f"Failed to cache orderbook: {e}")

        # Notify callbacks (outside lock)
        for callback in self._on_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(market_id, price, orderbook)
                else:
                    callback(market_id, price, orderbook)
            except Exception as e:
                logger.error(f"Error in price update callback: {e}")

    # --- Query Methods ---

    def get_current_price(self, market_id: str) -> Optional[float]:
        """Get the latest price for a market."""
        return self._market_prices.get(market_id)

    def get_current_orderbook(self, market_id: str) -> Optional[Orderbook]:
        """Get the latest orderbook for a market."""
        return self._market_orderbooks.get(market_id)

    def get_price_change(
        self,
        market_id: str,
        lookback_seconds: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get (old_price, new_price) for a lookback period.

        Args:
            market_id: The market to check
            lookback_seconds: How far back to look

        Returns:
            Tuple of (price_at_lookback, current_price) or (None, None) if unavailable
        """
        history = self._price_history.get(market_id)
        if not history:
            return None, None

        current_price = self._market_prices.get(market_id)
        if current_price is None:
            return None, None

        now = time.time()
        target_time = now - lookback_seconds

        # Find price closest to target_time
        old_price = None

        # History is ordered by time (oldest first due to deque)
        for point in history:
            if point.timestamp <= target_time:
                old_price = point.price
            else:
                # Found first point after target, use previous or this one
                if old_price is None:
                    old_price = point.price
                break

        # If we never found a point before target, use oldest available
        if old_price is None and history:
            old_price = history[0].price

        return old_price, current_price

    def get_price_history(
        self,
        market_id: str,
        lookback_seconds: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Get price history as list of (timestamp, price) tuples.

        Args:
            market_id: The market to query
            lookback_seconds: Optional limit on history (default: all available)

        Returns:
            List of (timestamp, price) tuples, oldest first
        """
        history = self._price_history.get(market_id)
        if not history:
            return []

        if lookback_seconds is None:
            return [(p.timestamp, p.price) for p in history]

        cutoff = time.time() - lookback_seconds
        return [(p.timestamp, p.price) for p in history if p.timestamp >= cutoff]

    def get_subscribed_markets(self) -> Set[str]:
        """Get the set of currently subscribed market IDs."""
        return self._subscribed_markets.copy()

    def get_all_current_prices(self) -> Dict[str, float]:
        """Get all current prices as a dictionary."""
        return self._market_prices.copy()

    # --- Manual Update (for testing/polling) ---

    def manual_price_update(
        self,
        market_id: str,
        price: float,
        orderbook: Optional[Orderbook] = None
    ) -> None:
        """
        Manually push a price update (for testing or REST API polling).

        This is synchronous and doesn't trigger async callbacks.
        """
        now = time.time()

        self._market_prices[market_id] = price
        if orderbook:
            self._market_orderbooks[market_id] = orderbook

        if market_id not in self._price_history:
            self._price_history[market_id] = deque(maxlen=self.max_history_points)

        self._price_history[market_id].append(PricePoint(
            timestamp=now,
            price=price,
            orderbook=orderbook
        ))

        # Prune old history
        cutoff = now - self.history_window_seconds
        while (self._price_history[market_id] and
               self._price_history[market_id][0].timestamp < cutoff):
            self._price_history[market_id].popleft()

        # Notify sync callbacks only
        for callback in self._on_update_callbacks:
            if not asyncio.iscoroutinefunction(callback):
                try:
                    callback(market_id, price, orderbook)
                except Exception as e:
                    logger.error(f"Error in sync callback: {e}")

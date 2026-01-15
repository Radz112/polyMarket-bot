"""
Polymarket WebSocket Client for real-time orderbook updates.

Provides a robust WebSocket connection with:
- Auto-reconnection with exponential backoff
- Multiple callback support per event type
- Message queuing during reconnection
- Connection state tracking
- Comprehensive logging
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import aiohttp

from src.api.exceptions import (
    WebSocketError,
    WebSocketConnectionError,
    SubscriptionError,
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


class MessageType(Enum):
    """Types of messages received from WebSocket."""
    BOOK = "book"
    PRICE_CHANGE = "price_change"
    LAST_TRADE_PRICE = "last_trade_price"
    TICK_SIZE_CHANGE = "tick_size_change"
    UNKNOWN = "unknown"


@dataclass
class ReconnectionPolicy:
    """
    Defines the reconnection behavior for the WebSocket client.
    
    Uses exponential backoff with optional jitter to prevent
    thundering herd problems when reconnecting.
    
    Attributes:
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        multiplier: Factor to multiply delay after each attempt
        max_attempts: Maximum number of reconnection attempts (0 = infinite)
        jitter: Whether to add random jitter to delays
    """
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    max_attempts: int = 10
    jitter: bool = True
    
    # Internal state
    _current_delay: float = field(default=1.0, init=False, repr=False)
    _attempts: int = field(default=0, init=False, repr=False)
    
    def __post_init__(self):
        self._current_delay = self.initial_delay
    
    def get_next_delay(self) -> float:
        """
        Get the next reconnection delay.
        
        Returns:
            Delay in seconds before next reconnection attempt
        """
        import random
        
        delay = self._current_delay
        
        # Add jitter (±25%)
        if self.jitter:
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)
        
        # Update for next call
        self._current_delay = min(
            self._current_delay * self.multiplier,
            self.max_delay
        )
        self._attempts += 1
        
        return delay
    
    def reset(self) -> None:
        """Reset the policy after successful connection."""
        self._current_delay = self.initial_delay
        self._attempts = 0
    
    def should_retry(self) -> bool:
        """
        Check if we should attempt reconnection.
        
        Returns:
            True if more attempts are allowed
        """
        if self.max_attempts == 0:
            return True  # Infinite retries
        return self._attempts < self.max_attempts
    
    @property
    def attempts(self) -> int:
        """Number of reconnection attempts made."""
        return self._attempts


# Callback type aliases
OrderbookCallback = Callable[[str, Dict[str, Any]], None]
PriceCallback = Callable[[str, float], None]
TradeCallback = Callable[[str, Dict[str, Any]], None]
ErrorCallback = Callable[[Exception], None]
DisconnectCallback = Callable[[], None]
RawMessageCallback = Callable[[Dict[str, Any]], None]


class PolymarketWebSocket:
    """
    WebSocket client for Polymarket real-time market data.
    
    Provides automatic reconnection, multiple event callbacks,
    and message queuing during disconnections.
    
    Example:
        ws = PolymarketWebSocket()
        
        @ws.on_orderbook_update
        def handle_orderbook(token_id: str, data: dict):
            print(f"Orderbook update for {token_id}")
        
        await ws.connect()
        await ws.subscribe_to_market("token_id_123")
        await ws.start()
    
    Attributes:
        url: WebSocket server URL
        state: Current connection state
        subscriptions: Set of subscribed token IDs
    """
    
    DEFAULT_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    def __init__(
        self,
        url: str = DEFAULT_URL,
        reconnection_policy: Optional[ReconnectionPolicy] = None,
        heartbeat_interval: float = 30.0,
        message_queue_size: int = 1000,
    ):
        """
        Initialize the WebSocket client.
        
        Args:
            url: WebSocket server URL
            reconnection_policy: Policy for handling reconnections
            heartbeat_interval: Interval for heartbeat/ping messages
            message_queue_size: Max size of message queue during reconnection
        """
        self.url = url
        self.reconnection_policy = reconnection_policy or ReconnectionPolicy()
        self.heartbeat_interval = heartbeat_interval
        self.message_queue_size = message_queue_size
        
        # Connection state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._state = ConnectionState.DISCONNECTED
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        
        # Subscriptions
        self._subscriptions: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        
        # Callbacks
        self._orderbook_callbacks: List[OrderbookCallback] = []
        self._price_callbacks: List[PriceCallback] = []
        self._trade_callbacks: List[TradeCallback] = []
        self._error_callbacks: List[ErrorCallback] = []
        self._disconnect_callbacks: List[DisconnectCallback] = []
        self._raw_callbacks: List[RawMessageCallback] = []
        
        # Message queue for messages received while reconnecting
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=message_queue_size)
        
        # Statistics
        self._messages_received = 0
        self._messages_processed = 0
        self._last_message_time: Optional[float] = None
        self._connect_time: Optional[float] = None
        
        logger.info(f"PolymarketWebSocket initialized: url={url}")
    
    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state
    
    @property
    def subscriptions(self) -> Set[str]:
        """Set of currently subscribed token IDs."""
        return self._subscriptions.copy()
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return (
            self._state == ConnectionState.CONNECTED and
            self._ws is not None and
            not self._ws.closed
        )
    
    @property
    def running(self) -> bool:
        """Check if the client is running."""
        return self._running
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        
        Raises:
            WebSocketConnectionError: If connection fails
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return
        
        self._state = ConnectionState.CONNECTING
        
        try:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=None, connect=30)
                self._session = aiohttp.ClientSession(timeout=timeout)
            
            self._ws = await self._session.ws_connect(
                self.url,
                heartbeat=self.heartbeat_interval,
                autoping=True,
            )
            
            self._state = ConnectionState.CONNECTED
            self._connect_time = time.time()
            self.reconnection_policy.reset()
            
            logger.info(f"Connected to {self.url}")
            
            # Resubscribe to previous subscriptions
            if self._subscriptions:
                await self._resubscribe()
                
        except aiohttp.ClientError as e:
            self._state = ConnectionState.DISCONNECTED
            logger.error(f"Failed to connect: {e}")
            raise WebSocketConnectionError(f"Connection failed: {e}")
        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            logger.error(f"Unexpected error during connect: {e}")
            raise WebSocketConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close connection gracefully."""
        if self._state == ConnectionState.DISCONNECTED:
            return
        
        self._state = ConnectionState.CLOSING
        self._running = False
        
        # Cancel listen task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._ws = None
        self._session = None
        self._state = ConnectionState.DISCONNECTED
        
        logger.info(
            f"Disconnected after receiving {self._messages_received} messages"
        )
    
    async def reconnect(self) -> bool:
        """
        Reconnect after disconnection.
        
        Returns:
            True if reconnection succeeded, False otherwise
        """
        if not self.reconnection_policy.should_retry():
            logger.error(
                f"Max reconnection attempts ({self.reconnection_policy.max_attempts}) reached"
            )
            return False
        
        self._state = ConnectionState.RECONNECTING
        delay = self.reconnection_policy.get_next_delay()
        
        logger.info(
            f"Reconnecting in {delay:.1f}s "
            f"(attempt {self.reconnection_policy.attempts})"
        )
        
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
            return True
        except WebSocketConnectionError:
            return False
    
    async def _resubscribe(self) -> None:
        """Resubscribe to all previous subscriptions after reconnect."""
        if not self._subscriptions:
            return
        
        token_ids = list(self._subscriptions)
        logger.info(f"Resubscribing to {len(token_ids)} markets")
        
        try:
            await self._send_subscription("subscribe", token_ids)
        except Exception as e:
            logger.error(f"Failed to resubscribe: {e}")
    
    # =========================================================================
    # Subscriptions
    # =========================================================================
    
    async def subscribe_to_market(self, token_id: str) -> None:
        """
        Subscribe to a single market's updates.
        
        Args:
            token_id: Token ID to subscribe to
        """
        await self.subscribe_to_markets([token_id])
    
    async def subscribe_to_markets(self, token_ids: List[str]) -> None:
        """
        Subscribe to multiple markets.
        
        Args:
            token_ids: List of token IDs to subscribe to
        """
        new_ids = [tid for tid in token_ids if tid not in self._subscriptions]
        
        if not new_ids:
            logger.debug("All token IDs already subscribed")
            return
        
        # Add to subscriptions set
        for tid in new_ids:
            self._subscriptions.add(tid)
        
        # Send subscription if connected
        if self.is_connected:
            await self._send_subscription("subscribe", new_ids)
        else:
            # Queue for when we connect
            for tid in new_ids:
                self._pending_subscriptions.add(tid)
            logger.debug(f"Queued {len(new_ids)} subscriptions for when connected")
    
    async def unsubscribe_from_market(self, token_id: str) -> None:
        """
        Unsubscribe from a market.
        
        Args:
            token_id: Token ID to unsubscribe from
        """
        if token_id not in self._subscriptions:
            return
        
        self._subscriptions.discard(token_id)
        self._pending_subscriptions.discard(token_id)
        
        if self.is_connected:
            await self._send_subscription("unsubscribe", [token_id])
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all markets."""
        if not self._subscriptions:
            return
        
        token_ids = list(self._subscriptions)
        self._subscriptions.clear()
        self._pending_subscriptions.clear()
        
        if self.is_connected:
            await self._send_subscription("unsubscribe", token_ids)
    
    async def _send_subscription(
        self,
        operation: str,
        token_ids: List[str]
    ) -> None:
        """Send subscription/unsubscription message."""
        if not self._ws or self._ws.closed:
            raise SubscriptionError(f"Cannot {operation}: not connected")
        
        # Polymarket uses "assets_ids" in their subscription format
        payload = {
            "type": "MARKET",
            "operation": operation,
            "assets_ids": token_ids,
        }
        
        try:
            await self._ws.send_json(payload)
            logger.info(f"{operation.capitalize()}d to {len(token_ids)} markets")
        except Exception as e:
            logger.error(f"Failed to {operation}: {e}")
            raise SubscriptionError(f"Failed to {operation}: {e}")
    
    # =========================================================================
    # Event Callbacks
    # =========================================================================
    
    def on_orderbook_update(
        self,
        callback: OrderbookCallback
    ) -> OrderbookCallback:
        """
        Register callback for orderbook updates.
        
        Can be used as a decorator:
            @ws.on_orderbook_update
            def handle(token_id, data):
                ...
        
        Or called directly:
            ws.on_orderbook_update(my_handler)
        
        Args:
            callback: Function(token_id: str, data: dict) -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._orderbook_callbacks.append(callback)
        return callback
    
    def on_price_change(self, callback: PriceCallback) -> PriceCallback:
        """
        Register callback for price changes.
        
        Args:
            callback: Function(token_id: str, price: float) -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._price_callbacks.append(callback)
        return callback
    
    def on_trade(self, callback: TradeCallback) -> TradeCallback:
        """
        Register callback for trades.
        
        Args:
            callback: Function(token_id: str, trade_data: dict) -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._trade_callbacks.append(callback)
        return callback
    
    def on_error(self, callback: ErrorCallback) -> ErrorCallback:
        """
        Register callback for errors.
        
        Args:
            callback: Function(exception: Exception) -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._error_callbacks.append(callback)
        return callback
    
    def on_disconnect(self, callback: DisconnectCallback) -> DisconnectCallback:
        """
        Register callback for disconnections.
        
        Args:
            callback: Function() -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._disconnect_callbacks.append(callback)
        return callback
    
    def on_raw_message(self, callback: RawMessageCallback) -> RawMessageCallback:
        """
        Register callback for all raw messages.
        
        Args:
            callback: Function(data: dict) -> None
        
        Returns:
            The callback (for decorator use)
        """
        self._raw_callbacks.append(callback)
        return callback
    
    def remove_callback(self, callback: Callable) -> bool:
        """
        Remove a callback from all event types.
        
        Args:
            callback: The callback to remove
        
        Returns:
            True if callback was found and removed
        """
        removed = False
        for callback_list in [
            self._orderbook_callbacks,
            self._price_callbacks,
            self._trade_callbacks,
            self._error_callbacks,
            self._disconnect_callbacks,
            self._raw_callbacks,
        ]:
            if callback in callback_list:
                callback_list.remove(callback)
                removed = True
        return removed
    
    # =========================================================================
    # Message Handling
    # =========================================================================
    
    async def _handle_message(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Process a received message (or list of messages) and dispatch to callbacks."""
        # Polymarket sometimes sends a list of messages
        if isinstance(data, list):
            for item in data:
                await self._process_single_message(item)
        else:
            await self._process_single_message(data)
    
    async def _process_single_message(self, data: Dict[str, Any]) -> None:
        """Process a single message dictionary."""
        self._messages_received += 1
        self._last_message_time = time.time()
        
        # Fire raw callbacks
        await self._fire_callbacks(self._raw_callbacks, data)
        
        # Determine message type and dispatch
        msg_type = self._get_message_type(data)
        
        if msg_type == MessageType.BOOK:
            await self._handle_orderbook(data)
        elif msg_type == MessageType.PRICE_CHANGE:
            await self._handle_price_change(data)
        elif msg_type == MessageType.LAST_TRADE_PRICE:
            await self._handle_trade(data)
        else:
            logger.debug(f"Unhandled message type: {data.get('type', 'unknown')}")
        
        self._messages_processed += 1
    
    def _get_message_type(self, data: Dict[str, Any]) -> MessageType:
        """Determine the message type from data."""
        type_str = data.get("event_type", data.get("type", "")).lower()
        
        try:
            return MessageType(type_str)
        except ValueError:
            return MessageType.UNKNOWN
    
    async def _handle_orderbook(self, data: Dict[str, Any]) -> None:
        """Handle orderbook update message."""
        token_id = data.get("market", data.get("asset_id", ""))
        
        orderbook_data = {
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
            "timestamp": data.get("timestamp"),
            "hash": data.get("hash"),
        }
        
        await self._fire_callbacks(
            self._orderbook_callbacks,
            token_id,
            orderbook_data
        )
    
    async def _handle_price_change(self, data: Dict[str, Any]) -> None:
        """Handle price change message."""
        token_id = data.get("market", data.get("asset_id", ""))
        price_str = data.get("price", "0")
        
        try:
            price = float(price_str)
        except (ValueError, TypeError):
            price = 0.0
        
        await self._fire_callbacks(
            self._price_callbacks,
            token_id,
            price
        )
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade message."""
        token_id = data.get("market", data.get("asset_id", ""))
        
        trade_data = {
            "price": data.get("price"),
            "size": data.get("size"),
            "side": data.get("side"),
            "timestamp": data.get("timestamp"),
        }
        
        await self._fire_callbacks(
            self._trade_callbacks,
            token_id,
            trade_data
        )
    
    async def _fire_callbacks(
        self,
        callbacks: List[Callable],
        *args
    ) -> None:
        """Fire all callbacks with error handling."""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
                await self._fire_error(e)
    
    async def _fire_error(self, error: Exception) -> None:
        """Fire error callbacks."""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    async def _fire_disconnect(self) -> None:
        """Fire disconnect callbacks."""
        for callback in self._disconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    
    async def start(self) -> None:
        """
        Start listening for messages.
        
        This will run indefinitely until stop() is called.
        Handles reconnection automatically.
        """
        if self._running:
            logger.warning("Already running")
            return
        
        self._running = True
        
        # Connect if not already connected
        if not self.is_connected:
            await self.connect()
        
        # Start listening
        self._listen_task = asyncio.create_task(self._listen_loop())
        
        try:
            await self._listen_task
        except asyncio.CancelledError:
            logger.info("Listen task cancelled")
    
    async def _listen_loop(self) -> None:
        """Main message listening loop with reconnection."""
        while self._running:
            try:
                await self._receive_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                
                logger.error(f"Error in listen loop: {e}")
                await self._fire_error(e)
                await self._fire_disconnect()
                
                # Attempt reconnection
                if self._running:
                    success = await self.reconnect()
                    if not success and not self.reconnection_policy.should_retry():
                        logger.error("Giving up on reconnection")
                        self._running = False
                        break
    
    async def _receive_messages(self) -> None:
        """Receive and process messages from WebSocket."""
        if not self._ws:
            raise WebSocketConnectionError("Not connected")
        
        async for msg in self._ws:
            if not self._running:
                break
            
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                    
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.warning(f"WebSocket closed: {msg.data}")
                raise WebSocketConnectionError("Connection closed")
                
            elif msg.type == aiohttp.WSMsgType.ERROR:
                error = self._ws.exception() if self._ws else None
                logger.error(f"WebSocket error: {error}")
                raise WebSocketConnectionError(f"WebSocket error: {error}")
    
    async def stop(self) -> None:
        """Stop listening for messages."""
        logger.info("Stopping WebSocket client...")
        await self.disconnect()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dictionary with stats including messages received, uptime, etc.
        """
        uptime = None
        if self._connect_time and self.is_connected:
            uptime = time.time() - self._connect_time
        
        return {
            "state": self._state.value,
            "is_connected": self.is_connected,
            "subscriptions": len(self._subscriptions),
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "last_message_time": self._last_message_time,
            "uptime_seconds": uptime,
            "reconnect_attempts": self.reconnection_policy.attempts,
        }
    
    async def __aenter__(self) -> "PolymarketWebSocket":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
    
    def __repr__(self) -> str:
        return (
            f"PolymarketWebSocket(state={self._state.value}, "
            f"subscriptions={len(self._subscriptions)})"
        )


# For backward compatibility, alias to old name
ClobWsClient = PolymarketWebSocket

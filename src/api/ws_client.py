import asyncio
import json
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Callable, Set
from src.api.exceptions import WebSocketConnectionError, SubscriptionError

logger = logging.getLogger(__name__)

class ClobWsClient:
    """
    Asynchronous WebSocket client for Polymarket CLOB real-time market data.
    """
    def __init__(self, url: Optional[str] = None, asset_ids: Optional[List[str]] = None):
        self.url = url or "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.asset_ids = asset_ids or []
        self.callbacks: List[Callable[[Dict[str, Any]], Any]] = []
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

    def register_callback(self, callback: Callable[[Dict[str, Any]], Any]):
        """Registers a callback for market updates."""
        self.callbacks.append(callback)

    async def connect(self):
        """Establishes the WebSocket connection and starts the listening loop."""
        self._is_running = True
        while self._is_running:
            try:
                await self._run_socket()
            except Exception as e:
                if not self._is_running:
                    break
                logger.error(f"WebSocket error: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _run_socket(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)

        try:
            async with self._session.ws_connect(self.url, heartbeat=30) as ws:
                self._ws = ws
                self._reconnect_delay = 1
                logger.info(f"Connected to Polymarket WebSocket: {self.url}")
                
                if self.asset_ids:
                    await self.subscribe(self.asset_ids)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode WebSocket message: {e}")
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed")
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
        finally:
            self._ws = None

    async def _handle_message(self, data: Dict[str, Any]):
        """Dispatches messages to registered callbacks."""
        # Polymarket WS messages often have a 'event_type' or similar
        # For now, we pass all messages to callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")

    async def subscribe(self, asset_ids: List[str]):
        """Subscribes to market updates for the given asset IDs."""
        if not self._ws or self._ws.closed:
            # Update internal list if not connected, will be used on next connect
            for aid in asset_ids:
                if aid not in self.asset_ids:
                    self.asset_ids.append(aid)
            return

        payload = {
            "type": "MARKET",
            "operation": "subscribe",
            "assets_ids": asset_ids
        }
        await self._ws.send_json(payload)
        logger.info(f"Subscribed to assets: {asset_ids}")
        
        # Track for reconnections
        for aid in asset_ids:
            if aid not in self.asset_ids:
                self.asset_ids.append(aid)

    async def unsubscribe(self, asset_ids: List[str]):
        """Unsubscribes from market updates for the given asset IDs."""
        if self._ws and not self._ws.closed:
            payload = {
                "type": "MARKET",
                "operation": "unsubscribe",
                "assets_ids": asset_ids
            }
            await self._ws.send_json(payload)
            logger.info(f"Unsubscribed from assets: {asset_ids}")

        for aid in asset_ids:
            if aid in self.asset_ids:
                self.asset_ids.remove(aid)

    async def stop(self):
        """Stops the WebSocket client."""
        self._is_running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("WebSocket client stopped")

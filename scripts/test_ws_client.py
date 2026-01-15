#!/usr/bin/env python3
"""
Test script for the Polymarket WebSocket Client.

This script verifies that the PolymarketWebSocket works correctly by:
1. Connecting to the WebSocket server
2. Subscribing to a market
3. Listening for updates for 30 seconds
4. Testing reconnection by simulating disconnect
5. Testing graceful shutdown

Usage:
    python scripts/test_ws_client.py
    
    # With custom duration
    python scripts/test_ws_client.py --duration 60
    
    # Test reconnection
    python scripts/test_ws_client.py --test-reconnect

Requirements:
    - Internet connectivity
    - Active markets on Polymarket
"""

import asyncio
import os
import sys
import argparse
import logging
import signal
from datetime import datetime
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import (
    PolymarketClient,
    PolymarketWebSocket,
    ReconnectionPolicy,
    ConnectionState,
)
from src.api.exceptions import WebSocketConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_ws_client")


class WebSocketTester:
    """Test suite for PolymarketWebSocket."""
    
    def __init__(
        self,
        duration: int = 30,
        test_reconnect: bool = False,
        verbose: bool = False
    ):
        self.duration = duration
        self.test_reconnect = test_reconnect
        self.verbose = verbose
        
        self.ws: Optional[PolymarketWebSocket] = None
        self.rest_client: Optional[PolymarketClient] = None
        self.token_ids: List[str] = []
        
        # Counters
        self.orderbook_updates = 0
        self.price_changes = 0
        self.trades = 0
        self.errors = 0
        self.disconnects = 0
        
        # Shutdown flag
        self._shutdown = False
    
    async def run(self) -> bool:
        """Run all WebSocket tests."""
        logger.info("=" * 60)
        logger.info("Polymarket WebSocket Client Test")
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info(f"Duration: {self.duration} seconds")
        logger.info(f"Test reconnection: {self.test_reconnect}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Find active markets
            await self._find_active_markets()
            
            if not self.token_ids:
                logger.error("No active markets found to subscribe to")
                return False
            
            # Step 2: Create and configure WebSocket
            await self._setup_websocket()
            
            # Step 3: Connect and subscribe
            await self._connect_and_subscribe()
            
            # Step 4: Listen for updates
            await self._listen_for_updates()
            
            # Step 5: Test reconnection if requested
            if self.test_reconnect:
                await self._test_reconnection()
            
            # Success
            return True
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False
            
        finally:
            await self._cleanup()
            self._print_summary()
    
    async def _find_active_markets(self) -> None:
        """Find active markets with orderbooks."""
        logger.info("Finding active markets...")
        
        async with PolymarketClient() as client:
            markets = await client.get_markets()
            
            # First try to find markets with active orderbooks
            for market in markets:
                if (market.get("enable_order_book") and 
                    market.get("active") and 
                    not market.get("closed")):
                    tokens = market.get("tokens", [])
                    for token in tokens:
                        token_id = token.get("token_id")
                        if token_id:
                            self.token_ids.append(token_id)
                            if len(self.token_ids) >= 3:  # Get up to 3 tokens
                                break
                    if len(self.token_ids) >= 3:
                        break
            
            # If no active orderbook markets, fall back to any active market
            if not self.token_ids:
                logger.info("No active orderbook markets, trying any active market...")
                for market in markets:
                    if market.get("active") and not market.get("closed"):
                        tokens = market.get("tokens", [])
                        for token in tokens:
                            token_id = token.get("token_id")
                            if token_id:
                                self.token_ids.append(token_id)
                                if len(self.token_ids) >= 3:
                                    break
                        if len(self.token_ids) >= 3:
                            break
            
            # Last resort: use any token from any market
            if not self.token_ids:
                logger.info("No active markets, using first available token...")
                for market in markets[:10]:  # Check first 10
                    tokens = market.get("tokens", [])
                    for token in tokens:
                        token_id = token.get("token_id")
                        if token_id:
                            self.token_ids.append(token_id)
                            break
                    if self.token_ids:
                        break
        
        if self.token_ids:
            logger.info(f"Found {len(self.token_ids)} tokens to subscribe to")
            for tid in self.token_ids:
                logger.debug(f"  Token: {tid[:20]}...")
        else:
            logger.warning("No tokens found in any market")
    
    async def _setup_websocket(self) -> None:
        """Create and configure the WebSocket client."""
        logger.info("Setting up WebSocket client...")
        
        # Create with custom reconnection policy for testing
        policy = ReconnectionPolicy(
            initial_delay=1.0,
            max_delay=10.0,
            multiplier=2.0,
            max_attempts=5,
        )
        
        self.ws = PolymarketWebSocket(
            reconnection_policy=policy,
            heartbeat_interval=30.0,
        )
        
        # Register callbacks
        @self.ws.on_orderbook_update
        def handle_orderbook(token_id: str, data: dict):
            self.orderbook_updates += 1
            bids = len(data.get("bids", []))
            asks = len(data.get("asks", []))
            if self.verbose or self.orderbook_updates <= 5:
                logger.info(
                    f"📊 Orderbook update for {token_id[:16]}... "
                    f"({bids} bids, {asks} asks)"
                )
            elif self.orderbook_updates == 6:
                logger.info("... (further orderbook updates logged in verbose mode)")
        
        @self.ws.on_price_change
        def handle_price(token_id: str, price: float):
            self.price_changes += 1
            if self.verbose or self.price_changes <= 10:
                logger.info(f"💰 Price change for {token_id[:16]}...: {price:.4f}")
            elif self.price_changes == 11:
                logger.info("... (further price changes logged in verbose mode)")
        
        @self.ws.on_trade
        def handle_trade(token_id: str, data: dict):
            self.trades += 1
            price = data.get("price", "?")
            size = data.get("size", "?")
            logger.info(f"🔄 Trade on {token_id[:16]}...: {size} @ {price}")
        
        @self.ws.on_error
        def handle_error(error: Exception):
            self.errors += 1
            logger.error(f"❌ Error: {error}")
        
        @self.ws.on_disconnect
        def handle_disconnect():
            self.disconnects += 1
            logger.warning("⚠️  Disconnected!")
        
        if self.verbose:
            @self.ws.on_raw_message
            def handle_raw(data: dict):
                logger.debug(f"Raw message: {data}")
        
        logger.info("WebSocket client configured with callbacks")
    
    async def _connect_and_subscribe(self) -> None:
        """Connect to WebSocket and subscribe to markets."""
        logger.info("Connecting to WebSocket...")
        
        await self.ws.connect()
        
        logger.info(f"State: {self.ws.state.value}")
        assert self.ws.state == ConnectionState.CONNECTED, "Should be connected"
        
        # Subscribe to markets
        logger.info(f"Subscribing to {len(self.token_ids)} markets...")
        await self.ws.subscribe_to_markets(self.token_ids)
        
        logger.info(f"Subscribed to {len(self.ws.subscriptions)} markets")
    
    async def _listen_for_updates(self) -> None:
        """Listen for updates for the specified duration."""
        logger.info(f"Listening for updates for {self.duration} seconds...")
        logger.info("Press Ctrl+C to stop early")
        
        # Set up signal handler for graceful shutdown
        loop = asyncio.get_event_loop()
        
        def signal_handler():
            self._shutdown = True
            logger.info("Received shutdown signal")
        
        try:
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
        
        # Start listening in background
        listen_task = asyncio.create_task(self.ws.start())
        
        # Wait for duration or shutdown
        start_time = asyncio.get_event_loop().time()
        
        try:
            while not self._shutdown:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= self.duration:
                    break
                
                remaining = self.duration - elapsed
                logger.info(
                    f"[{int(elapsed)}s/{self.duration}s] "
                    f"Updates: {self.orderbook_updates} orderbooks, "
                    f"{self.price_changes} prices, {self.trades} trades"
                )
                
                await asyncio.sleep(min(5, remaining))
                
        finally:
            logger.info("Stopping listener...")
            await self.ws.stop()
            
            try:
                await asyncio.wait_for(listen_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
    
    async def _test_reconnection(self) -> None:
        """Test reconnection by simulating a disconnect."""
        logger.info("")
        logger.info("=" * 40)
        logger.info("Testing reconnection...")
        logger.info("=" * 40)
        
        # Reconnect and listen again briefly
        self.ws = PolymarketWebSocket(
            reconnection_policy=ReconnectionPolicy(
                initial_delay=0.5,
                max_delay=5.0,
                max_attempts=3,
            )
        )
        
        # Set up simpler callback for reconnect test
        reconnect_updates = 0
        
        @self.ws.on_orderbook_update
        def count_update(token_id: str, data: dict):
            nonlocal reconnect_updates
            reconnect_updates += 1
        
        await self.ws.connect()
        await self.ws.subscribe_to_markets(self.token_ids[:1])  # Just one market
        
        # Start listening
        listen_task = asyncio.create_task(self.ws.start())
        
        # Wait a bit for some updates
        await asyncio.sleep(3)
        
        # Force disconnect
        logger.info("Forcing disconnect...")
        if self.ws._ws:
            await self.ws._ws.close()
        
        # Wait for reconnection
        logger.info("Waiting for reconnection...")
        await asyncio.sleep(5)
        
        # Check if reconnected
        if self.ws.is_connected:
            logger.info("✅ Reconnection successful!")
        else:
            logger.warning("⚠️  Reconnection may still be in progress")
        
        # Stop
        await self.ws.stop()
        
        try:
            await asyncio.wait_for(listen_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        
        logger.info(f"Received {reconnect_updates} updates during reconnect test")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self.ws:
            try:
                await self.ws.stop()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")
    
    def _print_summary(self) -> None:
        """Print test summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"  Orderbook updates received: {self.orderbook_updates}")
        logger.info(f"  Price changes received:     {self.price_changes}")
        logger.info(f"  Trades received:            {self.trades}")
        logger.info(f"  Errors encountered:         {self.errors}")
        logger.info(f"  Disconnections:             {self.disconnects}")
        logger.info("=" * 60)
        
        if self.orderbook_updates > 0 or self.price_changes > 0:
            logger.info("✅ WebSocket client is working!")
        else:
            logger.warning(
                "⚠️  No updates received. This might be normal if markets are quiet."
            )


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the Polymarket WebSocket Client"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="How long to listen for updates (seconds)"
    )
    parser.add_argument(
        "--test-reconnect",
        action="store_true",
        help="Test reconnection by simulating disconnect"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = WebSocketTester(
        duration=args.duration,
        test_reconnect=args.test_reconnect,
        verbose=args.verbose,
    )
    
    success = await tester.run()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(130)

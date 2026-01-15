"""
Unit tests for the Polymarket WebSocket client components.

Tests the ReconnectionPolicy, PolymarketWebSocket callbacks,
and message handling without requiring network connectivity.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.api.websocket import (
    PolymarketWebSocket,
    ReconnectionPolicy,
    ConnectionState,
    MessageType,
)
from src.api.exceptions import (
    WebSocketConnectionError,
    SubscriptionError,
)


class TestReconnectionPolicy:
    """Tests for the ReconnectionPolicy class."""
    
    def test_init_default_values(self):
        """Test default initialization."""
        policy = ReconnectionPolicy()
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.multiplier == 2.0
        assert policy.max_attempts == 10
        assert policy.jitter is True
    
    def test_init_custom_values(self):
        """Test custom initialization."""
        policy = ReconnectionPolicy(
            initial_delay=0.5,
            max_delay=30.0,
            multiplier=1.5,
            max_attempts=5,
            jitter=False,
        )
        assert policy.initial_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.multiplier == 1.5
        assert policy.max_attempts == 5
        assert policy.jitter is False
    
    def test_get_next_delay_without_jitter(self):
        """Test delay calculation without jitter."""
        policy = ReconnectionPolicy(
            initial_delay=1.0,
            multiplier=2.0,
            jitter=False,
        )
        
        assert policy.get_next_delay() == 1.0
        assert policy.get_next_delay() == 2.0
        assert policy.get_next_delay() == 4.0
        assert policy.get_next_delay() == 8.0
    
    def test_get_next_delay_respects_max(self):
        """Test that delay is capped at max_delay."""
        policy = ReconnectionPolicy(
            initial_delay=10.0,
            max_delay=15.0,
            multiplier=2.0,
            jitter=False,
        )
        
        assert policy.get_next_delay() == 10.0
        assert policy.get_next_delay() == 15.0  # Capped
        assert policy.get_next_delay() == 15.0  # Still capped
    
    def test_get_next_delay_with_jitter(self):
        """Test delay with jitter is within expected range."""
        policy = ReconnectionPolicy(
            initial_delay=10.0,
            jitter=True,
        )
        
        delay = policy.get_next_delay()
        # Should be within ±25% of 10.0
        assert 7.5 <= delay <= 12.5
    
    def test_reset(self):
        """Test reset clears attempts and delay."""
        policy = ReconnectionPolicy(initial_delay=1.0, jitter=False)
        
        # Make some attempts
        policy.get_next_delay()
        policy.get_next_delay()
        assert policy.attempts == 2
        
        # Reset
        policy.reset()
        assert policy.attempts == 0
        assert policy.get_next_delay() == 1.0  # Back to initial
    
    def test_should_retry_with_limit(self):
        """Test should_retry respects max_attempts."""
        policy = ReconnectionPolicy(max_attempts=3)
        
        assert policy.should_retry() is True
        policy.get_next_delay()
        assert policy.should_retry() is True
        policy.get_next_delay()
        assert policy.should_retry() is True
        policy.get_next_delay()
        assert policy.should_retry() is False  # 3 attempts made
    
    def test_should_retry_infinite(self):
        """Test should_retry with infinite attempts."""
        policy = ReconnectionPolicy(max_attempts=0)  # 0 = infinite
        
        for _ in range(100):
            assert policy.should_retry() is True
            policy.get_next_delay()
    
    def test_attempts_property(self):
        """Test attempts property tracks correctly."""
        policy = ReconnectionPolicy()
        
        assert policy.attempts == 0
        policy.get_next_delay()
        assert policy.attempts == 1
        policy.get_next_delay()
        assert policy.attempts == 2


class TestPolymarketWebSocket:
    """Tests for the PolymarketWebSocket class."""
    
    def test_init_default_values(self):
        """Test default initialization."""
        ws = PolymarketWebSocket()
        assert ws.url == PolymarketWebSocket.DEFAULT_URL
        assert ws.state == ConnectionState.DISCONNECTED
        assert ws.is_connected is False
        assert ws.running is False
        assert len(ws.subscriptions) == 0
    
    def test_init_custom_values(self):
        """Test custom initialization."""
        policy = ReconnectionPolicy(max_attempts=5)
        ws = PolymarketWebSocket(
            url="wss://custom.url",
            reconnection_policy=policy,
            heartbeat_interval=15.0,
        )
        assert ws.url == "wss://custom.url"
        assert ws.reconnection_policy.max_attempts == 5
        assert ws.heartbeat_interval == 15.0
    
    def test_callback_registration_orderbook(self):
        """Test orderbook callback registration."""
        ws = PolymarketWebSocket()
        
        @ws.on_orderbook_update
        def handler(token_id: str, data: dict):
            pass
        
        assert len(ws._orderbook_callbacks) == 1
        assert handler in ws._orderbook_callbacks
    
    def test_callback_registration_price(self):
        """Test price callback registration."""
        ws = PolymarketWebSocket()
        
        @ws.on_price_change
        def handler(token_id: str, price: float):
            pass
        
        assert len(ws._price_callbacks) == 1
        assert handler in ws._price_callbacks
    
    def test_callback_registration_trade(self):
        """Test trade callback registration."""
        ws = PolymarketWebSocket()
        
        @ws.on_trade
        def handler(token_id: str, data: dict):
            pass
        
        assert len(ws._trade_callbacks) == 1
        assert handler in ws._trade_callbacks
    
    def test_callback_registration_error(self):
        """Test error callback registration."""
        ws = PolymarketWebSocket()
        
        @ws.on_error
        def handler(error: Exception):
            pass
        
        assert len(ws._error_callbacks) == 1
    
    def test_callback_registration_disconnect(self):
        """Test disconnect callback registration."""
        ws = PolymarketWebSocket()
        
        @ws.on_disconnect
        def handler():
            pass
        
        assert len(ws._disconnect_callbacks) == 1
    
    def test_remove_callback(self):
        """Test callback removal."""
        ws = PolymarketWebSocket()
        
        def handler(token_id: str, data: dict):
            pass
        
        ws.on_orderbook_update(handler)
        assert len(ws._orderbook_callbacks) == 1
        
        removed = ws.remove_callback(handler)
        assert removed is True
        assert len(ws._orderbook_callbacks) == 0
    
    def test_remove_nonexistent_callback(self):
        """Test removing callback that doesn't exist."""
        ws = PolymarketWebSocket()
        
        def handler():
            pass
        
        removed = ws.remove_callback(handler)
        assert removed is False
    
    def test_subscriptions_tracking(self):
        """Test that subscriptions are tracked correctly."""
        ws = PolymarketWebSocket()
        
        # Manually add subscriptions (normally done via subscribe_to_markets)
        ws._subscriptions.add("token1")
        ws._subscriptions.add("token2")
        
        subs = ws.subscriptions
        assert len(subs) == 2
        assert "token1" in subs
        assert "token2" in subs
        
        # Ensure it's a copy
        subs.add("token3")
        assert "token3" not in ws._subscriptions
    
    def test_get_stats(self):
        """Test stats retrieval."""
        ws = PolymarketWebSocket()
        
        stats = ws.get_stats()
        
        assert stats["state"] == "disconnected"
        assert stats["is_connected"] is False
        assert stats["subscriptions"] == 0
        assert stats["messages_received"] == 0
        assert stats["messages_processed"] == 0
        assert stats["reconnect_attempts"] == 0
    
    def test_repr(self):
        """Test string representation."""
        ws = PolymarketWebSocket()
        ws._subscriptions.add("token1")
        
        repr_str = repr(ws)
        assert "disconnected" in repr_str
        assert "subscriptions=1" in repr_str


class TestMessageType:
    """Tests for MessageType enum."""
    
    def test_valid_message_types(self):
        """Test parsing valid message types."""
        assert MessageType("book") == MessageType.BOOK
        assert MessageType("price_change") == MessageType.PRICE_CHANGE
        assert MessageType("last_trade_price") == MessageType.LAST_TRADE_PRICE
    
    def test_unknown_message_type(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError):
            MessageType("invalid_type")


class TestConnectionState:
    """Tests for ConnectionState enum."""
    
    def test_state_values(self):
        """Test connection state values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.CLOSING.value == "closing"


class TestMessageHandling:
    """Tests for message handling logic."""
    
    @pytest.mark.asyncio
    async def test_handle_orderbook_message(self):
        """Test orderbook message dispatches to callback."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_orderbook_update
        def handler(token_id: str, data: dict):
            received.append((token_id, data))
        
        message = {
            "type": "book",
            "market": "token123",
            "bids": [{"price": "0.55", "size": "100"}],
            "asks": [{"price": "0.56", "size": "50"}],
            "timestamp": "2024-01-15T10:30:00Z",
        }
        
        await ws._handle_message(message)
        
        assert len(received) == 1
        token_id, data = received[0]
        assert token_id == "token123"
        assert len(data["bids"]) == 1
        assert len(data["asks"]) == 1
    
    @pytest.mark.asyncio
    async def test_handle_price_change_message(self):
        """Test price change message dispatches to callback."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_price_change
        def handler(token_id: str, price: float):
            received.append((token_id, price))
        
        message = {
            "type": "price_change",
            "market": "token456",
            "price": "0.75",
        }
        
        await ws._handle_message(message)
        
        assert len(received) == 1
        assert received[0] == ("token456", 0.75)
    
    @pytest.mark.asyncio
    async def test_handle_trade_message(self):
        """Test trade message dispatches to callback."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_trade
        def handler(token_id: str, data: dict):
            received.append((token_id, data))
        
        message = {
            "type": "last_trade_price",
            "market": "token789",
            "price": "0.60",
            "size": "25",
        }
        
        await ws._handle_message(message)
        
        assert len(received) == 1
        token_id, data = received[0]
        assert token_id == "token789"
        assert data["price"] == "0.60"
        assert data["size"] == "25"
    
    @pytest.mark.asyncio
    async def test_handle_raw_message(self):
        """Test raw message callback receives all messages."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_raw_message
        def handler(data: dict):
            received.append(data)
        
        message = {"type": "unknown", "data": "test"}
        
        await ws._handle_message(message)
        
        assert len(received) == 1
        assert received[0] == message
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test that callback errors are caught and don't break processing."""
        ws = PolymarketWebSocket()
        
        errors = []
        
        @ws.on_orderbook_update
        def bad_handler(token_id: str, data: dict):
            raise ValueError("Test error")
        
        @ws.on_error
        def error_handler(error: Exception):
            errors.append(error)
        
        message = {
            "type": "book",
            "market": "token123",
            "bids": [],
            "asks": [],
        }
        
        # Should not raise
        await ws._handle_message(message)
        
        # Error should have been caught
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    
    @pytest.mark.asyncio
    async def test_async_callback_support(self):
        """Test that async callbacks work correctly."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_orderbook_update
        async def async_handler(token_id: str, data: dict):
            await asyncio.sleep(0.01)  # Simulate async work
            received.append(token_id)
        
        message = {
            "type": "book",
            "market": "async_token",
            "bids": [],
            "asks": [],
        }
        
        await ws._handle_message(message)
        
        assert len(received) == 1
        assert received[0] == "async_token"
    
    @pytest.mark.asyncio
    async def test_handle_batched_messages(self):
        """Test handling a list of messages (batched)."""
        ws = PolymarketWebSocket()
        
        received = []
        
        @ws.on_price_change
        def handler(token_id: str, price: float):
            received.append((token_id, price))
        
        messages = [
            {
                "type": "price_change",
                "market": "token1",
                "price": "0.50",
            },
            {
                "type": "price_change",
                "market": "token2",
                "price": "0.60",
            }
        ]
        
        await ws._handle_message(messages)
        
        assert len(received) == 2
        assert received[0] == ("token1", 0.50)
        assert received[1] == ("token2", 0.60)

    @pytest.mark.asyncio
    async def test_messages_counted(self):
        """Test that message counters are updated."""
        ws = PolymarketWebSocket()
        
        assert ws._messages_received == 0
        assert ws._messages_processed == 0
        
        message = {"type": "book", "market": "token", "bids": [], "asks": []}
        
        await ws._handle_message(message)
        
        assert ws._messages_received == 1
        assert ws._messages_processed == 1


class TestSubscriptions:
    """Tests for subscription management."""
    
    @pytest.mark.asyncio
    async def test_subscribe_adds_to_set(self):
        """Test that subscribing adds to internal set."""
        ws = PolymarketWebSocket()
        
        # Without connection, subscriptions should be tracked
        await ws.subscribe_to_markets(["token1", "token2"])
        
        assert "token1" in ws._subscriptions
        assert "token2" in ws._subscriptions
    
    @pytest.mark.asyncio
    async def test_subscribe_to_market_single(self):
        """Test subscribing to a single market."""
        ws = PolymarketWebSocket()
        
        await ws.subscribe_to_market("token1")
        
        assert "token1" in ws._subscriptions
    
    @pytest.mark.asyncio
    async def test_duplicate_subscriptions_ignored(self):
        """Test that duplicate subscriptions are handled."""
        ws = PolymarketWebSocket()
        
        await ws.subscribe_to_market("token1")
        await ws.subscribe_to_market("token1")  # Duplicate
        
        assert len(ws._subscriptions) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_set(self):
        """Test that unsubscribing removes from set."""
        ws = PolymarketWebSocket()
        
        await ws.subscribe_to_market("token1")
        assert "token1" in ws._subscriptions
        
        await ws.unsubscribe_from_market("token1")
        assert "token1" not in ws._subscriptions
    
    @pytest.mark.asyncio
    async def test_unsubscribe_all(self):
        """Test unsubscribe all clears subscriptions."""
        ws = PolymarketWebSocket()
        
        await ws.subscribe_to_markets(["token1", "token2", "token3"])
        assert len(ws._subscriptions) == 3
        
        await ws.unsubscribe_all()
        assert len(ws._subscriptions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

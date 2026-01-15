"""
Polymarket API Client Package.

This package provides async clients for interacting with the Polymarket CLOB API,
including REST endpoints and WebSocket streams.

Example:
    from src.api import PolymarketClient, PolymarketWebSocket
    
    async with PolymarketClient() as client:
        markets = await client.get_markets()
        orderbook = await client.get_order_book(token_id)
    
    async with PolymarketWebSocket() as ws:
        await ws.subscribe_to_market(token_id)
        await ws.start()
"""

from src.api.client import PolymarketClient
from src.api.websocket import (
    PolymarketWebSocket,
    ReconnectionPolicy,
    ConnectionState,
    MessageType,
)
from src.api.exceptions import (
    PolymarketError,
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
    MarketNotFoundError,
    OrderError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    OrderCancellationError,
    WebSocketError,
    WebSocketConnectionError,
    SubscriptionError,
    ValidationError,
    ConfigurationError,
)
from src.api.rate_limiter import RateLimiter
from src.api.utils import (
    retry_with_backoff,
    with_retry,
    normalize_side,
    normalize_order_type,
    format_price,
    format_size,
    parse_orderbook_response,
    calculate_spread,
)

# Keep backward compatibility with the old client names
from src.api.clob_client import ClobClient
from src.api.websocket import PolymarketWebSocket as ClobWsClient

__all__ = [
    # Main clients
    "PolymarketClient",
    "PolymarketWebSocket",
    # WebSocket utilities
    "ReconnectionPolicy",
    "ConnectionState",
    "MessageType",
    # Legacy clients
    "ClobClient",
    "ClobWsClient",
    # Exceptions
    "PolymarketError",
    "PolymarketAPIError",
    "RateLimitError",
    "AuthenticationError",
    "MarketNotFoundError",
    "OrderError",
    "InsufficientBalanceError",
    "InvalidOrderError",
    "OrderNotFoundError",
    "OrderCancellationError",
    "WebSocketError",
    "WebSocketConnectionError",
    "SubscriptionError",
    "ValidationError",
    "ConfigurationError",
    # Rate limiter
    "RateLimiter",
    # Utilities
    "retry_with_backoff",
    "with_retry",
    "normalize_side",
    "normalize_order_type",
    "format_price",
    "format_size",
    "parse_orderbook_response",
    "calculate_spread",
]

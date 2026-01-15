"""
Custom exceptions for the Polymarket API client.

Provides a hierarchy of exceptions for different error types encountered
when interacting with the Polymarket CLOB API.
"""

from typing import Optional


class PolymarketError(Exception):
    """
    Base exception for all Polymarket bot errors.
    
    All other exceptions inherit from this class, making it easy
    to catch any Polymarket-related error.
    """
    pass


class PolymarketAPIError(PolymarketError):
    """
    Exception raised for errors returned by the Polymarket API.
    
    Attributes:
        status: HTTP status code
        message: Error message from the API
        response_data: Optional raw response data
    """
    
    def __init__(
        self,
        status: int,
        message: str,
        response_data: Optional[dict] = None
    ):
        self.status = status
        self.message = message
        self.response_data = response_data
        super().__init__(f"API Error {status}: {message}")


class RateLimitError(PolymarketAPIError):
    """
    Exception raised when API rate limits are hit.
    
    The client should wait and retry after receiving this error.
    Check the retry_after attribute for recommended wait time.
    """
    
    def __init__(
        self,
        status: int = 429,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None
    ):
        super().__init__(status, message)
        self.retry_after = retry_after


class AuthenticationError(PolymarketAPIError):
    """
    Exception raised for authentication-related errors.
    
    This includes invalid API keys, expired credentials,
    and missing authentication for protected endpoints.
    """
    pass


class MarketNotFoundError(PolymarketAPIError):
    """
    Exception raised when a market or token is not found.
    
    This typically occurs when using an invalid condition_id or token_id.
    """
    pass


class OrderError(PolymarketAPIError):
    """
    Exception raised when order submission fails.
    
    This is the base class for order-related errors.
    """
    pass


class InsufficientBalanceError(OrderError):
    """
    Exception raised when there are not enough funds for an order.
    
    Check available balance before submitting orders to avoid this error.
    """
    
    def __init__(
        self,
        status: int = 400,
        message: str = "Insufficient balance",
        available_balance: Optional[float] = None,
        required_amount: Optional[float] = None
    ):
        super().__init__(status, message)
        self.available_balance = available_balance
        self.required_amount = required_amount


class OrderNotFoundError(OrderError):
    """Exception raised when an order cannot be found."""
    
    def __init__(self, order_id: str):
        super().__init__(404, f"Order not found: {order_id}")
        self.order_id = order_id


class InvalidOrderError(OrderError):
    """
    Exception raised when order parameters are invalid.
    
    Check price, size, and other parameters before submitting.
    """
    pass


class OrderCancellationError(OrderError):
    """Exception raised when order cancellation fails."""
    pass


class WebSocketError(PolymarketError):
    """
    Base exception for WebSocket related errors.
    
    All WebSocket-specific errors inherit from this class.
    """
    pass


class WebSocketConnectionError(WebSocketError):
    """
    Exception raised for WebSocket connection failures.
    
    This includes connection refused, timeout, and other network issues.
    """
    pass


class SubscriptionError(WebSocketError):
    """
    Exception raised for subscription failures.
    
    Occurs when subscribing to or unsubscribing from channels fails.
    """
    pass


class ValidationError(PolymarketError):
    """
    Exception raised when input validation fails.
    
    Check the field and message for details on what validation failed.
    """
    
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error for '{field}': {message}")


class ConfigurationError(PolymarketError):
    """
    Exception raised for configuration-related errors.
    
    This includes missing environment variables, invalid config values, etc.
    """
    pass

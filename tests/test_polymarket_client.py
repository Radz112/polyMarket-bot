"""
Unit tests for the new Polymarket API client components.

Tests the rate limiter, utilities, PolymarketClient, and exception classes
without requiring network connectivity.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

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
from src.api.exceptions import (
    PolymarketError,
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
    OrderError,
    InsufficientBalanceError,
    OrderNotFoundError,
    ValidationError,
)


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_init_default_values(self):
        """Test default initialization."""
        limiter = RateLimiter()
        assert limiter.max_requests == 60
        assert limiter.window_seconds == 60.0
        assert limiter.get_remaining() == 60
    
    def test_init_custom_values(self):
        """Test custom initialization."""
        limiter = RateLimiter(max_requests_per_minute=100, window_seconds=30.0)
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 30.0
    
    def test_record_request(self):
        """Test recording requests decreases remaining count."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        assert limiter.get_remaining() == 10
        limiter.record_request()
        assert limiter.get_remaining() == 9
        
        for _ in range(5):
            limiter.record_request()
        assert limiter.get_remaining() == 4
    
    def test_reset(self):
        """Test reset clears recorded requests."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        for _ in range(5):
            limiter.record_request()
        assert limiter.get_remaining() == 5
        
        limiter.reset()
        assert limiter.get_remaining() == 10
    
    def test_get_wait_time_no_wait(self):
        """Test wait time is 0 when under limit."""
        limiter = RateLimiter(max_requests_per_minute=10)
        assert limiter.get_wait_time() == 0.0
        
        limiter.record_request()
        assert limiter.get_wait_time() == 0.0
    
    def test_get_wait_time_limit_reached(self):
        """Test wait time is positive when limit reached."""
        limiter = RateLimiter(max_requests_per_minute=3, window_seconds=10.0)
        
        for _ in range(3):
            limiter.record_request()
        
        wait_time = limiter.get_wait_time()
        assert wait_time > 0
        assert wait_time <= 10.0
    
    @pytest.mark.asyncio
    async def test_acquire_no_wait(self):
        """Test acquire doesn't wait when under limit."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        start = time.monotonic()
        await limiter.acquire()
        duration = time.monotonic() - start
        
        assert duration < 0.1  # Should be nearly instant
        assert limiter.get_remaining() == 9
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        async with limiter:
            pass
        
        assert limiter.get_remaining() == 9
    
    def test_repr(self):
        """Test string representation."""
        limiter = RateLimiter(max_requests_per_minute=10)
        repr_str = repr(limiter)
        assert "max_requests=10" in repr_str
        assert "remaining=10" in repr_str


class TestUtils:
    """Tests for utility functions."""
    
    def test_normalize_side_valid(self):
        """Test normalize_side with valid inputs."""
        assert normalize_side("BUY") == "BUY"
        assert normalize_side("SELL") == "SELL"
        assert normalize_side("buy") == "BUY"
        assert normalize_side("sell") == "SELL"
        assert normalize_side("  BUY  ") == "BUY"
    
    def test_normalize_side_invalid(self):
        """Test normalize_side with invalid inputs."""
        with pytest.raises(ValueError, match="Invalid side"):
            normalize_side("INVALID")
        
        with pytest.raises(ValueError):
            normalize_side("long")
    
    def test_normalize_order_type_valid(self):
        """Test normalize_order_type with valid inputs."""
        assert normalize_order_type("GTC") == "GTC"
        assert normalize_order_type("FOK") == "FOK"
        assert normalize_order_type("GTD") == "GTD"
        assert normalize_order_type("gtc") == "GTC"
    
    def test_normalize_order_type_invalid(self):
        """Test normalize_order_type with invalid inputs."""
        with pytest.raises(ValueError, match="Invalid order type"):
            normalize_order_type("LIMIT")
    
    def test_format_price_valid(self):
        """Test format_price with valid inputs."""
        assert format_price(0.5) == "0.5000"
        assert format_price(0.0) == "0.0000"
        assert format_price(1.0) == "1.0000"
        assert format_price(0.123456) == "0.1235"  # Rounded
        assert format_price(0.00001) == "0.0000"  # Rounded to 0
    
    def test_format_price_invalid(self):
        """Test format_price with invalid inputs."""
        with pytest.raises(ValueError, match="Price must be between"):
            format_price(-0.1)
        
        with pytest.raises(ValueError):
            format_price(1.1)
    
    def test_format_size_valid(self):
        """Test format_size with valid inputs."""
        assert format_size(100.0) == "100.00"
        assert format_size(0.01) == "0.01"
        assert format_size(1234.567) == "1234.57"
    
    def test_format_size_invalid(self):
        """Test format_size with invalid inputs."""
        with pytest.raises(ValueError, match="Size must be positive"):
            format_size(0)
        
        with pytest.raises(ValueError):
            format_size(-10)
    
    def test_parse_orderbook_response_empty(self):
        """Test parsing empty orderbook."""
        result = parse_orderbook_response({})
        assert result["bids"] == []
        assert result["asks"] == []
    
    def test_parse_orderbook_response_with_data(self):
        """Test parsing orderbook with data."""
        raw = {
            "bids": [
                {"price": "0.45", "size": "100"},
                {"price": "0.44", "size": "50"},
            ],
            "asks": [
                {"price": "0.55", "size": "100"},
                {"price": "0.56", "size": "50"},
            ],
            "timestamp": "2024-01-01T00:00:00Z",
            "hash": "abc123",
        }
        
        result = parse_orderbook_response(raw)
        
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0]["price"] == 0.45
        assert result["bids"][1]["price"] == 0.44  # Sorted descending
        assert result["asks"][0]["price"] == 0.55
        assert result["asks"][1]["price"] == 0.56  # Sorted ascending
        assert result["timestamp"] == "2024-01-01T00:00:00Z"
    
    def test_calculate_spread_with_data(self):
        """Test spread calculation with valid data."""
        orderbook = {
            "bids": [{"price": 0.45, "size": 100}],
            "asks": [{"price": 0.55, "size": 100}],
        }
        
        result = calculate_spread(orderbook)
        
        assert result["best_bid"] == 0.45
        assert result["best_ask"] == 0.55
        assert abs(result["spread"] - 0.10) < 0.0001
        assert abs(result["midpoint"] - 0.50) < 0.0001
        assert abs(result["spread_percent"] - 20.0) < 0.1  # 0.10 / 0.50 * 100
    
    def test_calculate_spread_empty(self):
        """Test spread calculation with empty orderbook."""
        result = calculate_spread({"bids": [], "asks": []})
        
        assert result["best_bid"] is None
        assert result["best_ask"] is None
        assert result["spread"] is None
        assert result["midpoint"] is None


class TestRetryLogic:
    """Tests for retry with backoff."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_try(self):
        """Test successful call on first attempt."""
        mock_func = AsyncMock(return_value="success")
        
        result = await retry_with_backoff(mock_func, max_retries=3)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful call after some failures."""
        mock_func = AsyncMock(side_effect=[
            RateLimitError(429, "Rate limited"),
            RateLimitError(429, "Rate limited"),
            "success",
        ])
        
        result = await retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01,  # Fast for testing
        )
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self):
        """Test that exception is raised after all retries fail."""
        mock_func = AsyncMock(side_effect=RateLimitError(429, "Rate limited"))
        
        with pytest.raises(RateLimitError):
            await retry_with_backoff(
                mock_func,
                max_retries=2,
                base_delay=0.01,
            )
        
        assert mock_func.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        """Test that authentication errors are not retried."""
        mock_func = AsyncMock(side_effect=AuthenticationError(401, "Unauthorized"))
        
        with pytest.raises(AuthenticationError):
            await retry_with_backoff(mock_func, max_retries=3)
        
        assert mock_func.call_count == 1  # No retries


class TestExceptions:
    """Tests for exception classes."""
    
    def test_polymarket_error(self):
        """Test base exception."""
        error = PolymarketError("Something went wrong")
        assert str(error) == "Something went wrong"
    
    def test_polymarket_api_error(self):
        """Test API error with status code."""
        error = PolymarketAPIError(400, "Bad request")
        assert error.status == 400
        assert error.message == "Bad request"
        assert "400" in str(error)
        assert "Bad request" in str(error)
    
    def test_rate_limit_error_with_retry_after(self):
        """Test rate limit error with retry after hint."""
        error = RateLimitError(429, "Rate limited", retry_after=30.0)
        assert error.status == 429
        assert error.retry_after == 30.0
    
    def test_insufficient_balance_error(self):
        """Test insufficient balance error with details."""
        error = InsufficientBalanceError(
            400,
            "Not enough funds",
            available_balance=50.0,
            required_amount=100.0,
        )
        assert error.available_balance == 50.0
        assert error.required_amount == 100.0
    
    def test_order_not_found_error(self):
        """Test order not found error."""
        error = OrderNotFoundError("order123")
        assert error.order_id == "order123"
        assert "order123" in str(error)
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("price", "Must be between 0 and 1")
        assert error.field == "price"
        assert "price" in str(error)
        assert "Must be between 0 and 1" in str(error)
    
    def test_exception_hierarchy(self):
        """Test that exceptions have correct inheritance."""
        assert issubclass(PolymarketAPIError, PolymarketError)
        assert issubclass(RateLimitError, PolymarketAPIError)
        assert issubclass(AuthenticationError, PolymarketAPIError)
        assert issubclass(OrderError, PolymarketAPIError)
        assert issubclass(InsufficientBalanceError, OrderError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

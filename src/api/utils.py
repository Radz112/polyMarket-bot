"""
Utility functions for the Polymarket API client.

Contains helper functions for retry logic, request building, and data transformation.
"""

import asyncio
import functools
import logging
import random
from typing import TypeVar, Callable, Any, Optional, Type, Tuple
from src.api.exceptions import (
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        RateLimitError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
    **kwargs
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exceptions that trigger a retry
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function call
    
    Raises:
        The last exception if all retries fail
    """
    last_exception: Optional[Exception] = None
    
    func_name = getattr(func, '__name__', repr(func))
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {func_name}: {e}"
                )
                raise
            
            # Calculate delay with exponential backoff
            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )
            
            # Add jitter (±25% of delay)
            if jitter:
                jitter_range = delay * 0.25
                delay = delay + random.uniform(-jitter_range, jitter_range)
                delay = max(0.1, delay)  # Minimum 100ms delay
            
            logger.warning(
                f"Attempt {attempt + 1} failed for {func_name}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
        except AuthenticationError:
            # Don't retry authentication errors
            raise
        except PolymarketAPIError as e:
            # Only retry on 5xx errors
            if hasattr(e, 'status') and 500 <= e.status < 600:
                last_exception = e
                if attempt == max_retries:
                    raise
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )
                logger.warning(
                    f"Server error (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise
    
    if last_exception:
        raise last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """
    Decorator to add retry logic to async functions.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    
    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                **kwargs
            )
        return wrapper
    return decorator


def normalize_side(side: str) -> str:
    """
    Normalize order side to uppercase format.
    
    Args:
        side: Order side (buy, sell, BUY, SELL, etc.)
    
    Returns:
        Normalized side string (BUY or SELL)
    
    Raises:
        ValueError: If side is not valid
    """
    side_upper = side.upper().strip()
    if side_upper not in ("BUY", "SELL"):
        raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
    return side_upper


def normalize_order_type(order_type: str) -> str:
    """
    Normalize order type to uppercase format.
    
    Args:
        order_type: Order type (GTC, FOK, GTD, etc.)
    
    Returns:
        Normalized order type string
    
    Raises:
        ValueError: If order type is not valid
    """
    valid_types = ("GTC", "FOK", "GTD")
    type_upper = order_type.upper().strip()
    if type_upper not in valid_types:
        raise ValueError(
            f"Invalid order type: {order_type}. "
            f"Must be one of {valid_types}"
        )
    return type_upper


def format_price(price: float) -> str:
    """
    Format price for API requests.
    
    Polymarket prices are between 0 and 1 with up to 4 decimal places.
    
    Args:
        price: Price as float (0.0 to 1.0)
    
    Returns:
        Formatted price string
    
    Raises:
        ValueError: If price is out of range
    """
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"Price must be between 0 and 1, got {price}")
    return f"{price:.4f}"


def format_size(size: float) -> str:
    """
    Format order size for API requests.
    
    Args:
        size: Order size in USDC
    
    Returns:
        Formatted size string
    
    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")
    return f"{size:.2f}"


def parse_orderbook_response(response: dict) -> dict:
    """
    Parse and normalize orderbook response.
    
    Args:
        response: Raw orderbook response from API
    
    Returns:
        Normalized orderbook with bid/ask lists as floats
    """
    def parse_level(level: dict) -> dict:
        return {
            "price": float(level.get("price", 0)),
            "size": float(level.get("size", 0)),
        }
    
    bids = [parse_level(b) for b in response.get("bids", [])]
    asks = [parse_level(a) for a in response.get("asks", [])]
    
    # Sort bids descending, asks ascending
    bids.sort(key=lambda x: x["price"], reverse=True)
    asks.sort(key=lambda x: x["price"])
    
    return {
        "bids": bids,
        "asks": asks,
        "timestamp": response.get("timestamp"),
        "hash": response.get("hash"),
    }


def calculate_spread(orderbook: dict) -> dict:
    """
    Calculate bid-ask spread from orderbook.
    
    Args:
        orderbook: Normalized orderbook dictionary
    
    Returns:
        Dictionary with spread metrics
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {
            "best_bid": None,
            "best_ask": None,
            "spread": None,
            "spread_percent": None,
            "midpoint": None,
        }
    
    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]
    spread = best_ask - best_bid
    midpoint = (best_bid + best_ask) / 2
    spread_percent = (spread / midpoint * 100) if midpoint > 0 else 0
    
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_percent": spread_percent,
        "midpoint": midpoint,
    }


def run_sync(coro):
    """
    Run an async coroutine synchronously.
    
    Used to wrap sync py-clob-client methods for async execution.
    
    Args:
        coro: Coroutine to execute
    
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.run(coro)
    else:
        # Already in async context, use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

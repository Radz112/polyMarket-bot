"""
Rate limiter for Polymarket API requests.

Implements a sliding window rate limiter to prevent exceeding API limits.
"""

import asyncio
import time
import logging
from typing import Deque
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter for API requests.
    
    Tracks requests in a sliding time window and blocks when the limit
    is reached, waiting until capacity becomes available.
    
    Attributes:
        max_requests: Maximum requests allowed in the window
        window_seconds: Size of the sliding window in seconds
    
    Example:
        limiter = RateLimiter(max_requests_per_minute=60)
        async with limiter:
            await make_request()
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        window_seconds: float = 60.0
    ):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests allowed per window
            window_seconds: Size of the sliding window in seconds
        """
        self.max_requests = max_requests_per_minute
        self.window_seconds = window_seconds
        self._request_times: Deque[float] = deque()
        self._lock = asyncio.Lock()
        
        logger.debug(
            f"RateLimiter initialized: {max_requests_per_minute} requests "
            f"per {window_seconds} seconds"
        )
    
    def _cleanup_old_requests(self) -> None:
        """Remove request timestamps outside the current window."""
        cutoff = time.monotonic() - self.window_seconds
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
    
    def get_remaining(self) -> int:
        """
        Get the number of remaining requests in the current window.
        
        Returns:
            Number of requests that can still be made
        """
        self._cleanup_old_requests()
        return max(0, self.max_requests - len(self._request_times))
    
    def get_wait_time(self) -> float:
        """
        Get how long to wait before a request can be made.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        self._cleanup_old_requests()
        if len(self._request_times) < self.max_requests:
            return 0.0
        
        # Wait until the oldest request expires
        oldest = self._request_times[0]
        wait_time = (oldest + self.window_seconds) - time.monotonic()
        return max(0.0, wait_time)
    
    def record_request(self) -> None:
        """Record that a request was made at the current time."""
        self._request_times.append(time.monotonic())
    
    async def acquire(self) -> None:
        """
        Wait if necessary before allowing a request.
        
        This method will block if the rate limit has been reached,
        waiting until there is capacity for another request.
        """
        async with self._lock:
            wait_time = self.get_wait_time()
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached, waiting {wait_time:.2f}s "
                    f"({self.get_remaining()} remaining)"
                )
                await asyncio.sleep(wait_time)
            
            self.record_request()
            
            remaining = self.get_remaining()
            if remaining < 10:
                logger.debug(f"Rate limit warning: {remaining} requests remaining")
    
    async def __aenter__(self) -> "RateLimiter":
        """Context manager entry - acquires rate limit slot."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass
    
    def reset(self) -> None:
        """Reset the rate limiter, clearing all recorded requests."""
        self._request_times.clear()
        logger.debug("RateLimiter reset")
    
    def __repr__(self) -> str:
        return (
            f"RateLimiter(max_requests={self.max_requests}, "
            f"window_seconds={self.window_seconds}, "
            f"remaining={self.get_remaining()})"
        )

import time
import functools
import asyncio
import logging
from .logger import logger

def log_timing(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__name__} completed", duration_ms=round(duration, 2))
            return result
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            # Depending on severity, we might want to log as ERROR, but let's stick to debug/error as requested
            logger.error(f"{func.__name__} failed", duration_ms=round(duration, 2), error=str(e))
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__name__} completed", duration_ms=round(duration, 2))
            return result
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error(f"{func.__name__} failed", duration_ms=round(duration, 2), error=str(e))
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

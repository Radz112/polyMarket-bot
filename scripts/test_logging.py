import asyncio
import os
from src.config import config
from src.logging import logger, setup_logging, log_timing

@log_timing
async def timed_operation(duration: float):
    logger.debug(f"Sleeping for {duration}s")
    await asyncio.sleep(duration)
    return "done"

@log_timing
def sync_operation():
    logger.info("Doing sync work")
    return "sync_done"

async def main():
    print("--- Testing Configuration ---")
    print(f"Env: {config.env}")
    print(f"DB Host: {config.postgres_host}")
    print(f"Log Level: {config.log_level}")
    
    print("\n--- Testing Logging ---")
    # Setup logger
    setup_logging(config)
    
    logger.info("This is an INFO message")
    logger.debug("This is a DEBUG message", user_id=123, action="test")
    logger.warning("This is a WARNING")
    logger.error("This is an ERROR", error_code=500)
    
    # Context
    ctx_logger = logger.with_context(request_id="req_xyz")
    ctx_logger.info("Message with context")
    
    # Specialized
    logger.trade("BUY", market="mkt_1", size=100)
    logger.signal("DIVERGENCE", market="mkt_1", score=85)
    
    print("\n--- Testing Timing Decorator ---")
    await timed_operation(0.1)
    sync_operation()
    
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(main())

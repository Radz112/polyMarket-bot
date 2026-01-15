import asyncio
import os
import logging
from src.config import config
from src.logging import logger, log_timing

async def test_performance_logging():
    # Performance timing
    @log_timing
    async def fast_op():
        await asyncio.sleep(0.01)
        return "done"

    @log_timing
    async def slow_op():
        await asyncio.sleep(0.2)
        raise ValueError("Oops too slow")

    await fast_op()
    try:
        await slow_op()
    except ValueError:
        pass

async def main():
    print(f"Environment: {config.env}")
    print(f"Log Level: {config.log_level}")
    print(f"Log File: {config.log_file}")
    
    logger.info("Starting logging verification")
    
    # 1. Basic logging
    logger.debug("Debug message (should appear in file)")
    logger.warning("Warning: rate limit 90%")
    
    # 2. Context logging
    with logger.context(request_id="12345", user="test_user"):
        logger.info("Processing request")
        logger.error("Something went wrong", extra={"details": "db_timeout"})
        
    # 3. Trade logging
    logger.trade(
        action="BUY",
        market_id="0xABC",
        side="YES",
        size=500.0,
        price=0.65,
        is_paper=True
    )
    
    # 4. Signal logging
    logger.signal(
        signal_type="DIVERGENCE",
        market_ids=["m1", "m2"],
        divergence=0.045,
        score=0.92
    )

    # 5. Performance logging
    await test_performance_logging()
    
    print("\nCheck logs/bot.log for JSON output.")

if __name__ == "__main__":
    asyncio.run(main())

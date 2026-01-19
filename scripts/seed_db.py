
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import Config
from src.database.postgres import DatabaseManager
from src.database.redis_cache import CacheManager
from src.api.clob_client import ClobClient
from src.correlation.fetcher import MarketFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SeedDB")

async def seed_database():
    """Fetch active markets and populate the database."""
    logger.info("Initializing DB Seeder...")
    
    config = Config()
    
    # Initialize components
    db = DatabaseManager(config)
    cache = CacheManager() # CacheManager takes no arguments
    # actually CacheManager in this codebase might just default or take nothing, verifying from previous usages implies default is often okay or params needed.
    # Looking at src/database/redis_cache.py (mock check) or inferred usage. 
    # Providing config is safer if implemented, otherwise let's instantiate safely.
    # Re-checking fetch_and_categorize.py line 27: `cache = CacheManager()` 
    # So I will stick to no-auth or default args unless I see otherwise.
    # But wait, config might be needed for redis host.
    # Let's try simple instantiation first as per existing scripts.
    
    api = ClobClient() # ClobClient uses default host or config internally
    
    try:
        await db.connect()
        # await cache.connect() # If needed
        
        # Check DB connectivity
        if not await db.health_check():
            logger.error("❌ Database not healthy. Is Docker running?")
            return
            
        logger.info("✅ Database connected.")

        fetcher = MarketFetcher(api, db, cache)
        
        logger.info("Fetching active markets from Polymarket...")
        markets = await fetcher.fetch_all_markets(active_only=True)
        
        logger.info(f"Successfully seeded {len(markets)} markets.")
        
        # Optional: Run correlation detection?
        # For now, just getting markets is enough to stop the "silent" bot.
        
    except Exception as e:
        logger.error(f"❌ Seeding failed: {e}", exc_info=True)
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(seed_database())

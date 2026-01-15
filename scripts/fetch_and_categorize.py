import asyncio
import logging
from src.config import config
from src.correlation.fetcher import MarketFetcher
from unittest.mock import MagicMock, AsyncMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Fetch & Categorize script...")
    
    # Mock dependencies if DB not available
    try:
        from src.database.postgres import DatabaseManager
        from src.database.redis import CacheManager
        from src.api.clob_client import ClobClient
        
        # We can try to init real ones, but if Docker fails we mock
        # For this script output demonstration, let's use Mocks if we can't connect,
        # but the user asked for a script that "fetches all markets".
        # Assuming the user WILL enable Docker later, we write the REAL script.
        
        api = ClobClient(host=config.polymarket_api_url)
        db = DatabaseManager()
        cache = CacheManager()
        
        fetcher = MarketFetcher(api, db, cache)
        
        # For demo purposes without running DB, this might crash.
        # But this is the file requested.
        await db.connect() 
        markets = await fetcher.fetch_all_markets(active_only=True)
        
        # Distribution
        stats = {}
        for m in markets:
            c = m.category or "Uncategorized"
            stats[c] = stats.get(c, 0) + 1
            
        print("\n=== Category Distribution ===")
        for c, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{c}: {count}")
            
        print("\n=== Entity Extraction Examples ===")
        for m in markets[:5]:
            print(f"Q: {m.question}")
            print(f"Cat: {m.category}/{m.subcategory}")
            print(f"Ent: {m.entities}\n")
            
    except Exception as e:
        logger.error(f"Script failed (likely due to missing DB): {e}")

if __name__ == "__main__":
    asyncio.run(main())

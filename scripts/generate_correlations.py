import asyncio
import logging
import time
from src.database.postgres import DatabaseManager
from src.correlation.store import CorrelationStore
from src.correlation.similarity.detector import StringSimilarityDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CorrelationGenerator")

async def main():
    logger.info("Starting Correlation Generation...")
    
    # 1. Initialize Database
    db = DatabaseManager()
    await db.connect()
    
    try:
        # 2. Fetch Active Markets
        logger.info("Fetching active markets from DB...")
        markets = await db.get_active_markets()
        logger.info(f"Loaded {len(markets)} active markets.")
        
        if not markets:
            logger.warning("No active markets found. Run seed_db.py first.")
            return

        # 3. Detect Correlations
        logger.info("Running similarity detection (this may take a moment)...")
        detector = StringSimilarityDetector(threshold=0.5) # Production threshold
        
        start_time = time.time()
        results = detector.find_similar_markets(markets)
        end_time = time.time()
        
        logger.info(f"Detection took {end_time - start_time:.2f}s")
        logger.info(f"Found {len(results)} potential correlations.")

        # 4. Save to Database
        if results:
            store = CorrelationStore(db)
            logger.info(f"Saving {len(results)} correlations to database...")
            await store.save_correlations_batch(results)
            logger.info("✅ Correlations saved successfully.")
        else:
            logger.info("No substantial correlations found.")
            
    except Exception as e:
        logger.error(f"Error during correlation generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

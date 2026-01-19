import asyncio
import logging
from typing import List, Optional

from src.api.clob_client import ClobClient
from src.database.postgres import DatabaseManager
from src.database.redis_cache import CacheManager
from src.models.market import Market
from .categorizer import MarketCategorizer

logger = logging.getLogger(__name__)

class MarketFetcher:
    def __init__(self, api_client: ClobClient, db: DatabaseManager, cache: CacheManager):
        self.api = api_client
        self.db = db
        self.cache = cache
        self.categorizer = MarketCategorizer()

    async def fetch_all_markets(self, active_only: bool = True) -> List[Market]:
        """Fetch all markets from API, categorize, and store."""
        logger.info("Fetching all markets...")
        # Note: Gamma API pagination logic might be needed for 'all' markets. 
        # ClobClient.get_markets needs to handle pagination or we loop here.
        # Assuming ClobClient returns a reasonable list or we implement pagination there later.
        # For now, let's assume we get a list.
        
        # TODO: ClobClient.get_markets might default to 100 or requires a next_cursor.
        # Check ClobClient implementation. It currently just passes params to generic request.
        # We will implement a simplified fetch for now.
        
        markets = await self.api.get_markets(limit=None if not active_only else 500) # Fetch chunk
        
        processed_markets = []
        for m_data in markets:
            if not isinstance(m_data, dict):
                continue
                
            try:
                # Map API fields to Market model
                # Note: Adjust field mapping based on actual API response structure
                m = Market(
                    condition_id=m_data.get("condition_id", ""),
                    slug=m_data.get("slug") or "",
                    question=m_data.get("question") or "",
                    description=m_data.get("description"),
                    end_date=m_data.get("end_date_iso"), 
                    active=m_data.get("active", True),
                    closed=m_data.get("closed", False),
                    resolved=m_data.get("resolved", False),
                    clob_token_ids=[t.get("token_id") for t in m_data.get("tokens", []) if isinstance(t, dict)]
                )
            except Exception as e:
                logger.debug(f"Skipping invalid market data: {e}")
                continue

            # Categorize
            cat, subcat, entities = self.categorizer.categorize(m)
            m.category = cat
            m.subcategory = subcat
            m.entities = entities
            
            # Save to DB
            await self.db.upsert_market(m)
            processed_markets.append(m)
            
        logger.info(f"Fetched and processed {len(processed_markets)} markets.")
        return processed_markets

    async def get_markets_by_category(self, category: str) -> List[Market]:
        return await self.db.get_markets_by_category(category)

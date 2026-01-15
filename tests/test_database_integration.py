import asyncio
import os
import pytest
from src.database.postgres import DatabaseManager
from src.database.redis import CacheManager
from src.models import Market, Orderbook, OrderbookEntry

# MARK: Integration tests requiring Docker
# These will be skipped if docker is not running or env vars not set, 
# but for now we assume environment is ready as per verify steps.

@pytest.mark.asyncio
async def test_postgres_crud():
    # 1. Connect
    db = DatabaseManager()
    await db.connect()
    assert await db.health_check() is True
    
    # 2. Cleanup (for test idempotency)
    # Be careful in real env, using DELETE for test market
    # In a real test we might rollback transaction or use a test DB
    
    # 3. Create Market
    m = Market(
        condition_id="int_test_1",
        slug="integration-test",
        question="Is this a test?",
        active=True
    )
    await db.upsert_market(m)
    
    # 4. Read Market
    fetched = await db.get_market("int_test_1")
    assert fetched is not None
    assert fetched.slug == "integration-test"
    
    # 5. Search
    results = await db.search_markets("Is this a test")
    assert len(results) >= 1
    assert results[0].id == "int_test_1"
    
    await db.disconnect()

@pytest.mark.asyncio
async def test_redis_cache():
    cache = CacheManager()
    # Assume default localhost:6379 works or from env
    try:
        await cache.connect()
    except:
        pytest.skip("Redis not available")
        
    if not await cache.health_check():
        pytest.skip("Redis health check failed")

    # 1. Set Orderbook
    ob = Orderbook(
        market_id="m123",
        bids=[OrderbookEntry(price=0.5, size=100)],
        asks=[OrderbookEntry(price=0.6, size=100)]
    )
    await cache.set_orderbook("m123", ob)
    
    # 2. Get Orderbook
    cached_ob = await cache.get_orderbook("m123")
    assert cached_ob is not None
    assert cached_ob.market_id == "m123"
    assert cached_ob.best_bid == 0.5
    
    await cache.disconnect()

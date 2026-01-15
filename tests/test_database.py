import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.postgres import DatabaseManager
from src.database.redis import CacheManager
from src.database.models import MarketModel
from src.models.market import Market
from src.models import Orderbook, OrderbookEntry, PriceSnapshot, Signal, SignalType

@pytest.mark.asyncio
async def test_get_market():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = MarketModel(
        id="123", slug="test", question="?", active=True, resolved=False, end_date=None, description=None, category=None, outcome=None
    )
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    
    manager = DatabaseManager()
    manager._session_factory = MagicMock(return_value=mock_session)
    
    market = await manager.get_market("123")
    assert market is not None
    assert market.id == "123"

@pytest.mark.asyncio
async def test_upsert_market():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    mock_session.execute.return_value = mock_result
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    
    manager = DatabaseManager()
    manager._session_factory = MagicMock(return_value=mock_session)
    
    market_in = Market(condition_id="new1", slug="new", question="Q")
    await manager.upsert_market(market_in)
    
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()

@pytest.mark.asyncio
async def test_cache_orderbook():
    manager = CacheManager()
    manager._redis = AsyncMock()
    
    ob = Orderbook(
        market_id="m1",
        bids=[OrderbookEntry(price=0.5, size=10)],
        asks=[]
    )
    await manager.set_orderbook("m1", ob)
    manager._redis.set.assert_called_once()

@pytest.mark.asyncio
async def test_cache_price():
    manager = CacheManager()
    manager._redis = AsyncMock()
    ps = PriceSnapshot(market_id="m1", yes_price=0.5, no_price=0.5)
    await manager.set_price("m1", ps)
    manager._redis.set.assert_called_once()

@pytest.mark.asyncio
async def test_signal_caching():
    manager = CacheManager()
    manager._redis = AsyncMock()
    
    sig = Signal(
        id="s1", 
        signal_type=SignalType.DIVERGENCE, 
        market_ids=["m1"], 
        divergence_amount=0.1, 
        expected_value=0.5, 
        actual_value=0.6, 
        confidence=0.9, 
        score=10.0
    )
    await manager.add_active_signal(sig)
    
    manager._redis.set.assert_called_once()
    manager._redis.zadd.assert_called_once()

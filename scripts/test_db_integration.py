import asyncio
import logging
import os
from datetime import datetime, timedelta
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_db_integration")

from src.models import (
    Market, Token, MarketCorrelation, CorrelationType,
    PriceSnapshot, Signal, SignalType,
    Position, Trade, Side, Orderbook
)
from src.database import DatabaseManager, CacheManager

async def test_postgres():
    logger.info("--- Testing PostgreSQL ---")
    db = DatabaseManager()
    
    if not await db.health_check():
        logger.warning("Postgres connection failed/unhealthy. Is docker running?")
        return

    await db.connect()
    
    try:
        # 1. Market
        market_id = f"test_market_{uuid.uuid4().hex[:8]}"
        market = Market(
            condition_id=market_id,
            slug="test-slug",
            question="Will this test pass?",
            active=True
        )
        logger.info(f"Upserting market {market_id}")
        await db.upsert_market(market)
        
        fetched = await db.get_market(market_id)
        assert fetched is not None
        assert fetched.id == market_id
        logger.info("Market verified")
        
        # 2. Price History
        snapshot = PriceSnapshot(
            market_id=market_id,
            token_id="t1",
            timestamp=datetime.utcnow(),
            yes_price=0.5,
            no_price=0.5,
            volume=100.0
        )
        logger.info("Saving price snapshot")
        await db.save_price_snapshot(snapshot)
        
        history = await db.get_price_history(market_id, datetime.utcnow() - timedelta(hours=1))
        assert len(history) >= 1
        logger.info(f"Price history verified: {len(history)} records")
        
        # 3. Signals
        sig_id = f"test_sig_{uuid.uuid4().hex[:8]}"
        signal = Signal(
            id=sig_id,
            signal_type=SignalType.DIVERGENCE,
            market_ids=[market_id],
            divergence_amount=0.1,
            expected_value=0.6,
            actual_value=0.5,
            confidence=0.9,
            score=80
        )
        logger.info("Saving signal")
        await db.save_signal(signal)
        
        recent = await db.get_recent_signals(limit=5)
        assert any(s.id == sig_id for s in recent)
        logger.info("Signals verified")

        # 4. Trades
        trade_id = f"test_trade_{uuid.uuid4().hex[:8]}"
        trade = Trade(
            id=trade_id,
            market_id=market_id,
            side=Side.YES,
            action="BUY",
            size=10.0,
            price=0.5,
            fees=0.01,
            timestamp=datetime.utcnow()
        )
        logger.info("Saving trade")
        await db.save_trade(trade)
        
        trades = await db.get_trades(market_id=market_id)
        assert len(trades) >= 1
        logger.info("Trades verified")
        
    finally:
        await db.disconnect()

async def test_redis():
    logger.info("--- Testing Redis ---")
    cache = CacheManager()
    
    try:
        await cache.connect()
        if not await cache.health_check():
            logger.warning("Redis connection unhealthy.")
            return

        # 1. Orderbook
        market_id = "test_market_ob"
        ob = Orderbook(
            market_id=market_id,
            token_id="t1"
        )
        logger.info("Setting orderbook")
        await cache.set_orderbook(market_id, ob)
        
        fetched_ob = await cache.get_orderbook(market_id)
        assert fetched_ob is not None
        assert fetched_ob.market_id == market_id
        logger.info("Orderbook verified")
        
        # 2. Signal
        sig_id = f"test_sig_redis_{uuid.uuid4().hex[:8]}"
        signal = Signal(
            id=sig_id,
            signal_type=SignalType.DIVERGENCE,
            market_ids=[market_id],
            divergence_amount=0.1,
            expected_value=0.6,
            actual_value=0.5,
            confidence=0.9,
            score=95
        )
        logger.info("Adding active signal")
        await cache.add_active_signal(signal)
        
        top = await cache.get_top_signals(limit=1)
        assert len(top) >= 1
        assert top[0].id == sig_id
        logger.info("Active signals verified")
        
    finally:
        await cache.disconnect()

async def main():
    await test_postgres()
    await test_redis()

if __name__ == "__main__":
    asyncio.run(main())

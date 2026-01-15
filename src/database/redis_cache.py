import logging
import os
import json
from datetime import timedelta
from typing import Optional, List, Dict, Callable, Union, Any
import redis.asyncio as redis
from src.models import (
    Orderbook, OrderbookUpdate, PriceSnapshot, Market, Signal
)

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None

    async def connect(self):
        """Initialize Redis connection."""
        if self._redis:
            return

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD", None)
        
        try:
            self._redis = redis.Redis(
                host=host, 
                port=port, 
                password=password, 
                decode_responses=True
            )
            await self._redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    # --- Orderbooks ---

    async def set_orderbook(self, market_id: str, orderbook: Orderbook, ttl: int = 60):
        if not self._redis: return
        key = f"orderbook:{market_id}"
        data = orderbook.model_dump_json()
        await self._redis.set(key, data, ex=ttl)

    async def get_orderbook(self, market_id: str) -> Optional[Orderbook]:
        if not self._redis: return None
        key = f"orderbook:{market_id}"
        data = await self._redis.get(key)
        if data:
            return Orderbook.model_validate_json(data)
        return None
    
    async def get_orderbooks(self, market_ids: List[str]) -> Dict[str, Orderbook]:
        if not self._redis or not market_ids: return {}
        keys = [f"orderbook:{mid}" for mid in market_ids]
        data = await self._redis.mget(keys)
        
        result = {}
        for market_id, value in zip(market_ids, data):
            if value:
                try:
                    result[market_id] = Orderbook.model_validate_json(value)
                except Exception as e:
                    logger.error(f"Error parsing orderbook for {market_id}: {e}")
        return result

    # --- Prices ---

    async def set_price(self, market_id: str, price: PriceSnapshot, ttl: int = 60):
        if not self._redis: return
        key = f"price:{market_id}"
        data = price.model_dump_json()
        await self._redis.set(key, data, ex=ttl)

    async def get_price(self, market_id: str) -> Optional[PriceSnapshot]:
        if not self._redis: return None
        key = f"price:{market_id}"
        data = await self._redis.get(key)
        if data:
            return PriceSnapshot.model_validate_json(data)
        return None

    async def get_prices(self, market_ids: List[str]) -> Dict[str, PriceSnapshot]:
        if not self._redis or not market_ids: return {}
        keys = [f"price:{mid}" for mid in market_ids]
        data = await self._redis.mget(keys)
        
        result = {}
        for market_id, value in zip(market_ids, data):
            if value:
                try:
                    result[market_id] = PriceSnapshot.model_validate_json(value)
                except Exception:
                    pass
        return result

    # --- Markets ---

    async def set_market(self, market: Market, ttl: int = 3600):
        if not self._redis: return
        key = f"market:{market.id}"
        data = market.model_dump_json()
        await self._redis.set(key, data, ex=ttl)

    async def get_market(self, market_id: str) -> Optional[Market]:
        if not self._redis: return None
        key = f"market:{market_id}"
        data = await self._redis.get(key)
        if data:
            return Market.model_validate_json(data)
        return None

    # --- Active Signals ---

    async def add_active_signal(self, signal: Signal):
        if not self._redis: return
        # 1. Store payload
        sig_key = f"signal:data:{signal.id}"
        await self._redis.set(sig_key, signal.model_dump_json(), ex=3600)
        
        # 2. Add to sorted set
        await self._redis.zadd("signal:active", {signal.id: signal.score})

    async def get_top_signals(self, limit: int = 10) -> List[Signal]:
        if not self._redis: return []
        ids = await self._redis.zrevrange("signal:active", 0, limit - 1)
        if not ids:
            return []
        
        keys = [f"signal:data:{sid}" for sid in ids]
        payloads = await self._redis.mget(keys)
        
        signals = []
        for p in payloads:
            if p:
                signals.append(Signal.model_validate_json(p))
        return signals

    async def remove_signal(self, signal_id: str):
        if not self._redis: return
        await self._redis.zrem("signal:active", signal_id)
        await self._redis.delete(f"signal:data:{signal_id}")

    # --- Pub/Sub ---

    async def publish(self, channel: str, message: dict):
        if not self._redis: return
        await self._redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str, callback: Callable[[dict], Any]):
        if not self._redis: return
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()
        
        await self._pubsub.subscribe(**{channel: callback})
        # Note: Subscribing with a callback usually requires a listening loop in async redis
        # The caller might need to perform run_in_thread or similar if using listen() directly
        # But redis-py async pubsub usually works by run_in_thread or iterating listen()
        pass 

    async def publish_orderbook_update(self, market_id: str, update: OrderbookUpdate):
        if not self._redis: return
        channel = f"updates:orderbook:{market_id}"
        await self._redis.publish(channel, update.model_dump_json())

    async def publish_signal(self, signal: Signal):
        if not self._redis: return
        channel = "updates:signals"
        await self._redis.publish(channel, signal.model_dump_json())

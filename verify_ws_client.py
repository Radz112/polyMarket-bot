import asyncio
import logging
import sys
import os
import aiohttp
from src.api.ws_client import ClobWsClient
from src.logging_config import setup_logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

async def get_live_token_id() -> str:
    """Fetches a live token ID from Gamma API for testing."""
    url = "https://gamma-api.polymarket.com/events?closed=false&limit=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if data and len(data) > 0:
                market = data[0]['markets'][0]
                tokens = market.get('clobTokenIds', [])
                if tokens:
                    try:
                        return eval(tokens)[0]
                    except:
                        pass
                return "21742461143292440317421045233516518171096700057088118029272332617157973041151"
    return "21742461143292440317421045233516518171096700057088118029272332617157973041151"

async def market_callback(data):
    logging.info(f"RECEIVED UPDATE: {data}")

async def main():
    setup_logging(logging.INFO)
    logger = logging.getLogger("WsVerifier")
    
    token_id = await get_live_token_id()
    logger.info(f"Starting WebSocket client for token: {token_id}")
    
    ws_client = ClobWsClient(asset_ids=[token_id])
    ws_client.register_callback(market_callback)
    
    # Run connection in background
    conn_task = asyncio.create_task(ws_client.connect())
    
    logger.info("Listening for 15 seconds...")
    try:
        await asyncio.sleep(15)
    finally:
        await ws_client.stop()
        conn_task.cancel()
        logger.info("Verification complete.")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
import sys
import os
import aiohttp
from src.api.clob_client import ClobClient
from src.logging_config import setup_logging

# Add current directory to sys.path to ensure src is importable
sys.path.append(os.getcwd())

async def get_live_token_id() -> str:
    """Fetches a live token ID from Gamma API for testing."""
    url = "https://gamma-api.polymarket.com/events?closed=false&limit=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if data and len(data) > 0:
                # Get the first token ID from the first market of the first event
                market = data[0]['markets'][0]
                # Return the 'token_id' (usually for 'Yes' or indexed)
                # CLOB uses token_id, but Gamma returns clobTokenIds or individual tokens
                tokens = market.get('clobTokenIds', [])
                if tokens:
                    return eval(tokens)[0] # It's often a string representation of a list
                return market.get('clobTokenIds') or "21742461143292440317421045233516518171096700057088118029272332617157973041151"
    return "21742461143292440317421045233516518171096700057088118029272332617157973041151"

async def main():
    setup_logging(logging.INFO)
    logger = logging.getLogger("Verifier")
    client = ClobClient()

    logger.info("--- Testing CLOB REST Client ---")

    try:
        # Step 1: Get a live token ID so the test actually works
        logger.info("Fetching a live token ID from Gamma API...")
        test_token_id = await get_live_token_id()
        logger.info(f"Using token ID: {test_token_id}")

        # Test 1: Get Midpoint
        logger.info(f"Testing get_midpoint...")
        midpoint = await client.get_midpoint(test_token_id)
        logger.info(f"SUCCESS: Midpoint: {midpoint}")

        # Test 2: Get Price
        logger.info(f"Testing get_price (buy)...")
        price = await client.get_price(test_token_id, side="buy")
        logger.info(f"SUCCESS: Price: {price}")

        # Test 3: Get Orderbook
        logger.info(f"Testing get_orderbook...")
        book = await client.get_orderbook(test_token_id)
        logger.info(f"SUCCESS: Orderbook fetched (bids: {len(book.get('bids', []))}, asks: {len(book.get('asks', []))})")

    except Exception as e:
        logger.error(f"FAILED: An error occurred: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

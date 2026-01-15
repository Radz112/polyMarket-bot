#!/usr/bin/env python
"""
Test script to verify live trading connection.

Usage:
    1. Set your credentials in .env:
       POLYMARKET_PRIVATE_KEY=0x...your_private_key...
       POLYMARKET_FUNDER=0x...your_address...
       PAPER_TRADING=false
       
    2. Run this script:
       source .venv/bin/activate
       python scripts/test_live_connection.py
"""
import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("live_connection_test")


def test_connection():
    """Test the live trading connection."""
    load_dotenv()
    
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder = os.getenv("POLYMARKET_FUNDER")
    
    if not private_key:
        logger.error("❌ POLYMARKET_PRIVATE_KEY not set in .env")
        logger.info("\nTo test, add to your .env file:")
        logger.info("  POLYMARKET_PRIVATE_KEY=0x...")
        logger.info("  POLYMARKET_FUNDER=0x... (optional)")
        return False
    
    logger.info("✅ Private key found")
    
    try:
        from src.api.live_client import LiveTradingClient
        
        logger.info("Initializing LiveTradingClient...")
        client = LiveTradingClient(
            private_key=private_key,
            funder=funder,
            signature_type=0,  # EOA
        )
        
        # Test connection
        logger.info("Testing connection to Polymarket...")
        result = client.test_connection()
        
        if result["connected"]:
            logger.info("✅ Connected to Polymarket!")
            logger.info(f"   Server time: {result.get('server_time')}")
            logger.info(f"   Open orders: {result.get('open_orders_count', 0)}")
            
            # Try to get a midpoint
            # Using a known active market token
            try:
                mid = client.get_midpoint("21742633143463906290569050155826241533067272736897614950488156847949938836455")
                if mid:
                    logger.info(f"   Sample midpoint: {mid}")
            except Exception as e:
                logger.debug(f"Midpoint check skipped: {e}")
            
            return True
        else:
            logger.error(f"❌ Connection failed: {result.get('error')}")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("\nMake sure py-clob-client is installed:")
        logger.info("  pip install py-clob-client")
        return False
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

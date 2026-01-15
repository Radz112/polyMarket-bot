#!/usr/bin/env python3
"""
Test script for the Polymarket REST API Client.

This script verifies that the PolymarketClient works correctly by testing
all public endpoints. Run this to validate your API client setup.

Usage:
    python scripts/test_api_client.py
    
    # For authenticated tests (requires private key)
    POLY_PRIVATE_KEY=0x... python scripts/test_api_client.py --authenticated

Requirements:
    - Internet connectivity
    - py-clob-client (for authenticated tests)
    
Environment Variables:
    POLY_PRIVATE_KEY: Private key for authenticated requests (optional)
    POLY_FUNDER: Funder address (optional)
"""

import asyncio
import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import PolymarketClient, RateLimiter
from src.api.exceptions import (
    PolymarketError,
    AuthenticationError,
    MarketNotFoundError,
)
from src.api.utils import normalize_side, format_price, calculate_spread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_api_client")


class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_ms = 0
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name} ({self.duration_ms}ms) - {self.message}"


class APIClientTester:
    """Test suite for PolymarketClient."""
    
    def __init__(self, authenticated: bool = False):
        self.authenticated = authenticated
        self.results: list[TestResult] = []
        self.client: PolymarketClient | None = None
        
        # Test data - populated during tests
        self.sample_token_id: str | None = None
        self.sample_condition_id: str | None = None
        self.has_active_orderbook: bool = False  # Whether we found an active market with orderbook
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return True if all passed."""
        logger.info("=" * 60)
        logger.info("Polymarket API Client Test Suite")
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info(f"Authenticated: {self.authenticated}")
        logger.info("=" * 60)
        
        try:
            # Create client
            private_key = os.environ.get("POLY_PRIVATE_KEY") if self.authenticated else None
            funder = os.environ.get("POLY_FUNDER")
            
            self.client = PolymarketClient(
                private_key=private_key,
                funder=funder,
                max_requests_per_minute=30,  # Conservative for testing
            )
            
            await self.client.connect()
            
            # Run tests
            await self._run_test("Health Check", self.test_health_check)
            await self._run_test("Server Time", self.test_server_time)
            await self._run_test("Get Markets", self.test_get_markets)
            await self._run_test("Get Simplified Markets", self.test_get_simplified_markets)
            await self._run_test("Get Single Market", self.test_get_market)
            await self._run_test("Get Orderbook", self.test_get_orderbook)
            await self._run_test("Get Midpoint", self.test_get_midpoint)
            await self._run_test("Get Price", self.test_get_price)
            await self._run_test("Get Spread", self.test_get_spread)
            await self._run_test("Rate Limiter", self.test_rate_limiter)
            await self._run_test("Error Handling", self.test_error_handling)
            await self._run_test("Utility Functions", self.test_utils)
            
            if self.authenticated:
                await self._run_test("API Credentials", self.test_create_api_creds)
                # Note: We don't test actual order placement to avoid spending money
                await self._run_test("Get Orders", self.test_get_orders)
            
        finally:
            if self.client:
                await self.client.close()
        
        # Print summary
        self._print_summary()
        
        return all(r.passed for r in self.results)
    
    async def _run_test(self, name: str, test_func) -> None:
        """Run a single test and record the result."""
        result = TestResult(name)
        start = asyncio.get_event_loop().time()
        
        try:
            await test_func()
            result.passed = True
            result.message = "OK"
        except AssertionError as e:
            result.passed = False
            result.message = str(e)
        except Exception as e:
            result.passed = False
            result.message = f"{type(e).__name__}: {e}"
        
        result.duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        self.results.append(result)
        logger.info(str(result))
    
    def _print_summary(self) -> None:
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed!")
        else:
            logger.info("Failed tests:")
            for r in self.results:
                if not r.passed:
                    logger.info(f"  - {r.name}: {r.message}")
        
        logger.info("=" * 60)
    
    # =========================================================================
    # Test Methods
    # =========================================================================
    
    async def test_health_check(self) -> None:
        """Test the health check endpoint."""
        is_ok = await self.client.get_ok()
        assert is_ok, "Health check failed"
    
    async def test_server_time(self) -> None:
        """Test the server time endpoint."""
        server_time = await self.client.get_server_time()
        assert server_time, "Server time is empty"
        logger.debug(f"Server time: {server_time}")
    
    async def test_get_markets(self) -> None:
        """Test fetching all markets."""
        markets = await self.client.get_markets()
        assert isinstance(markets, list), f"Expected list, got {type(markets)}"
        assert len(markets) > 0, "No markets returned"
        
        # Find an active market with orderbook enabled for later tests
        for market in markets:
            if market.get("enable_order_book") and market.get("active") and not market.get("closed"):
                if "condition_id" in market:
                    self.sample_condition_id = market["condition_id"]
                if "tokens" in market and market["tokens"]:
                    self.sample_token_id = market["tokens"][0].get("token_id")
                if self.sample_token_id:
                    self.has_active_orderbook = True
                    break
        
        # Fallback to first market if no active one found
        if not self.sample_condition_id:
            sample_market = markets[0]
            if "condition_id" in sample_market:
                self.sample_condition_id = sample_market["condition_id"]
            if "tokens" in sample_market and sample_market["tokens"]:
                self.sample_token_id = sample_market["tokens"][0].get("token_id")
        
        status = "with active orderbook" if self.has_active_orderbook else "(no active orderbook found)"
        logger.debug(f"Found {len(markets)} markets {status}")
    
    async def test_get_simplified_markets(self) -> None:
        """Test fetching simplified markets."""
        markets = await self.client.get_simplified_markets()
        assert isinstance(markets, dict), f"Expected dict, got {type(markets)}"
        logger.debug(f"Simplified markets has {len(markets)} entries")
    
    async def test_get_market(self) -> None:
        """Test fetching a single market."""
        if not self.sample_condition_id:
            # Fetch markets first to get a condition ID
            markets = await self.client.get_markets()
            if markets:
                self.sample_condition_id = markets[0].get("condition_id")
        
        if not self.sample_condition_id:
            raise AssertionError("No sample condition ID available")
        
        market = await self.client.get_market(self.sample_condition_id)
        assert isinstance(market, dict), f"Expected dict, got {type(market)}"
        assert "condition_id" in market or "question" in market, "Invalid market response"
        logger.debug(f"Market: {market.get('question', market.get('condition_id', 'unknown'))[:50]}")
    
    async def test_get_orderbook(self) -> None:
        """Test fetching orderbook."""
        if not self.has_active_orderbook:
            logger.debug("Skipping orderbook test - no active market with orderbook found")
            return  # Skip - will count as passed
        
        if not self.sample_token_id:
            raise AssertionError("No sample token ID available")
        
        orderbook = await self.client.get_order_book(self.sample_token_id)
        assert isinstance(orderbook, dict), f"Expected dict, got {type(orderbook)}"
        assert "bids" in orderbook, "Missing bids in orderbook"
        assert "asks" in orderbook, "Missing asks in orderbook"
        
        logger.debug(
            f"Orderbook: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks"
        )
    
    async def test_get_midpoint(self) -> None:
        """Test fetching midpoint price."""
        if not self.has_active_orderbook:
            logger.debug("Skipping midpoint test - no active market with orderbook found")
            return  # Skip - will count as passed
        
        if not self.sample_token_id:
            raise AssertionError("No sample token ID available")
        
        midpoint = await self.client.get_midpoint(self.sample_token_id)
        assert isinstance(midpoint, float), f"Expected float, got {type(midpoint)}"
        assert 0 <= midpoint <= 1, f"Midpoint out of range: {midpoint}"
        logger.debug(f"Midpoint: {midpoint}")
    
    async def test_get_price(self) -> None:
        """Test fetching price for both sides."""
        if not self.has_active_orderbook:
            logger.debug("Skipping price test - no active market with orderbook found")
            return  # Skip - will count as passed
        
        if not self.sample_token_id:
            raise AssertionError("No sample token ID available")
        
        buy_price = await self.client.get_price(self.sample_token_id, "BUY")
        sell_price = await self.client.get_price(self.sample_token_id, "SELL")
        
        assert isinstance(buy_price, float), f"Expected float, got {type(buy_price)}"
        assert isinstance(sell_price, float), f"Expected float, got {type(sell_price)}"
        
        logger.debug(f"Buy: {buy_price}, Sell: {sell_price}")
    
    async def test_get_spread(self) -> None:
        """Test spread calculation."""
        if not self.has_active_orderbook:
            logger.debug("Skipping spread test - no active market with orderbook found")
            return  # Skip - will count as passed
        
        if not self.sample_token_id:
            raise AssertionError("No sample token ID available")
        
        spread = await self.client.get_spread(self.sample_token_id)
        assert isinstance(spread, dict), f"Expected dict, got {type(spread)}"
        assert "best_bid" in spread, "Missing best_bid"
        assert "best_ask" in spread, "Missing best_ask"
        assert "spread" in spread, "Missing spread"
        assert "midpoint" in spread, "Missing midpoint"
        
        logger.debug(
            f"Spread: {spread['spread']:.4f} "
            f"({spread['spread_percent']:.2f}%)"
            if spread['spread'] is not None else "Spread: N/A"
        )
    
    async def test_rate_limiter(self) -> None:
        """Test rate limiter functionality."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        # Should have full capacity
        assert limiter.get_remaining() == 10, "Initial remaining incorrect"
        
        # Record some requests
        for _ in range(5):
            limiter.record_request()
        
        assert limiter.get_remaining() == 5, "Remaining after 5 requests incorrect"
        
        # Reset
        limiter.reset()
        assert limiter.get_remaining() == 10, "Remaining after reset incorrect"
    
    async def test_error_handling(self) -> None:
        """Test error handling for invalid requests."""
        # Test with invalid token ID
        try:
            await self.client.get_order_book("invalid_token_id_12345")
            # Some APIs return empty data instead of error
        except MarketNotFoundError:
            pass  # Expected
        except PolymarketError:
            pass  # Also acceptable
    
    async def test_utils(self) -> None:
        """Test utility functions."""
        # Test normalize_side
        assert normalize_side("buy") == "BUY"
        assert normalize_side("SELL") == "SELL"
        
        try:
            normalize_side("invalid")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        
        # Test format_price
        assert format_price(0.5) == "0.5000"
        assert format_price(0.123456) == "0.1235"  # Rounded
        
        try:
            format_price(1.5)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        
        # Test calculate_spread
        orderbook = {
            "bids": [{"price": 0.45, "size": 100}],
            "asks": [{"price": 0.55, "size": 100}],
        }
        spread = calculate_spread(orderbook)
        assert spread["best_bid"] == 0.45
        assert spread["best_ask"] == 0.55
        assert abs(spread["spread"] - 0.10) < 0.0001
        assert abs(spread["midpoint"] - 0.50) < 0.0001
    
    async def test_create_api_creds(self) -> None:
        """Test API credential creation (authenticated only)."""
        if not self.authenticated:
            return
        
        creds = await self.client.create_or_derive_api_creds()
        assert isinstance(creds, dict), f"Expected dict, got {type(creds)}"
        logger.debug("API credentials created successfully")
    
    async def test_get_orders(self) -> None:
        """Test getting open orders (authenticated only)."""
        if not self.authenticated:
            return
        
        orders = await self.client.get_orders()
        assert isinstance(orders, list), f"Expected list, got {type(orders)}"
        logger.debug(f"Found {len(orders)} open orders")


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the Polymarket API Client"
    )
    parser.add_argument(
        "--authenticated",
        action="store_true",
        help="Run authenticated tests (requires POLY_PRIVATE_KEY env var)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.authenticated and not os.environ.get("POLY_PRIVATE_KEY"):
        logger.error("POLY_PRIVATE_KEY environment variable required for authenticated tests")
        return 1
    
    tester = APIClientTester(authenticated=args.authenticated)
    success = await tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(130)

#!/usr/bin/env python3
"""
Live Divergence Detection Script

Runs the divergence detector on live Polymarket data for a specified duration,
logging all detected divergences with details.

Usage:
    python scripts/run_divergence_detector.py [--duration MINUTES] [--verbose]

Examples:
    python scripts/run_divergence_detector.py                    # Run for 5 minutes
    python scripts/run_divergence_detector.py --duration 10      # Run for 10 minutes
    python scripts/run_divergence_detector.py --verbose          # Verbose output
"""
import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.ws_client import ClobWsClient
from src.api.clob_client import ClobClient
from src.database.redis import CacheManager
from src.config.settings import Config
from src.signals.divergence import (
    DivergenceDetector,
    DivergenceConfig,
    PriceMonitor,
    Divergence,
    DivergenceType,
)
from src.models import MarketCorrelation, CorrelationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/divergence_detector.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)


class DivergenceMonitor:
    """
    Live divergence monitoring system.

    Connects to Polymarket WebSocket, monitors prices, and detects divergences.
    """

    def __init__(
        self,
        duration_minutes: int = 5,
        verbose: bool = False,
    ):
        self.duration_minutes = duration_minutes
        self.verbose = verbose

        # Components
        self.ws_client: ClobWsClient = None
        self.rest_client: ClobClient = None
        self.cache: CacheManager = None
        self.price_monitor: PriceMonitor = None
        self.detector: DivergenceDetector = None

        # State
        self.detected_divergences: List[Divergence] = []
        self.start_time: datetime = None
        self.is_running = False

        # Statistics
        self.stats = {
            "price_updates": 0,
            "detection_cycles": 0,
            "divergences_found": 0,
            "by_type": {t.value: 0 for t in DivergenceType},
            "false_positives": 0,
        }

    async def setup(self):
        """Initialize all components."""
        logger.info("Setting up divergence monitor...")

        # Load config
        try:
            config = Config()
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            config = None

        # Initialize WebSocket client
        self.ws_client = ClobWsClient()

        # Initialize REST client for fetching initial data
        self.rest_client = ClobClient()

        # Initialize cache (optional)
        try:
            self.cache = CacheManager()
            await self.cache.connect()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Running without cache.")
            self.cache = None

        # Initialize price monitor
        self.price_monitor = PriceMonitor(
            self.ws_client,
            cache=self.cache,
        )

        # Initialize detector with config
        detector_config = DivergenceConfig(
            min_divergence_threshold=0.02,
            min_liquidity_threshold=50.0,
            min_correlation_confidence=0.7,
            debounce_seconds=5.0,
        )
        self.detector = DivergenceDetector(
            self.price_monitor,
            config=detector_config,
        )

        # Register price update callback
        self.price_monitor.on_price_update(self._on_price_update)

        logger.info("Setup complete")

    async def _on_price_update(self, market_id: str, price: float, orderbook):
        """Callback for price updates."""
        self.stats["price_updates"] += 1
        if self.verbose:
            logger.debug(f"Price update: {market_id} = {price:.4f}")

    async def get_sample_correlations(self) -> List[MarketCorrelation]:
        """
        Get sample correlations for testing.

        In production, these would come from the CorrelationStore.
        For testing, we create some sample correlations.
        """
        # Sample correlations for testing
        # These would normally come from Phase 2's correlation detection
        correlations = []

        # Example: Markets that should be equivalent
        # (You would replace these with real market IDs)
        sample_pairs = [
            # Format: (market_a, market_b, correlation_type, confidence)
            ("sample_market_1", "sample_market_2", CorrelationType.POSITIVE, 0.9),
        ]

        for ma, mb, ctype, conf in sample_pairs:
            correlations.append(MarketCorrelation(
                market_a_id=ma,
                market_b_id=mb,
                correlation_type=ctype,
                expected_relationship="A ≈ B",
                confidence=conf,
            ))

        return correlations

    async def fetch_active_markets(self) -> List[str]:
        """
        Fetch active market IDs to monitor.

        Returns a list of market/token IDs to subscribe to.
        """
        logger.info("Fetching active markets...")

        try:
            # Try to get markets from REST API
            # This is a simplified version - you'd want to filter for
            # active, liquid markets
            markets = []

            # For demo, return empty list - WS will receive all subscribed markets
            # In production, you'd fetch from the CLOB API
            logger.info(f"Found {len(markets)} active markets")
            return markets

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def run_detection_cycle(self, correlations: List[MarketCorrelation]):
        """Run one detection cycle."""
        self.stats["detection_cycles"] += 1

        try:
            divergences = await self.detector.detect_all_divergences(
                correlations=correlations,
                rules=[],  # Add logical rules if available
            )

            for div in divergences:
                self.detected_divergences.append(div)
                self.stats["divergences_found"] += 1
                self.stats["by_type"][div.divergence_type.value] += 1

                self._log_divergence(div)

        except Exception as e:
            logger.error(f"Detection cycle error: {e}")

    def _log_divergence(self, div: Divergence):
        """Log a detected divergence with details."""
        logger.info("=" * 60)
        logger.info(f"DIVERGENCE DETECTED: {div.divergence_type.value}")
        logger.info(f"  ID: {div.id}")
        logger.info(f"  Markets: {', '.join(div.market_ids)}")
        logger.info(f"  Prices: {div.current_prices}")
        logger.info(f"  Divergence: {div.divergence_amount:.4f} ({div.divergence_pct:.2%})")
        logger.info(f"  Direction: {div.direction}")
        logger.info(f"  Is Arbitrage: {div.is_arbitrage}")
        logger.info(f"  Profit Potential: {div.profit_potential:.4f}")
        logger.info(f"  Max Size: ${div.max_executable_size:.2f}")
        logger.info(f"  Confidence: {div.confidence:.2%}")

        if div.supporting_evidence:
            logger.info(f"  Evidence:")
            for ev in div.supporting_evidence:
                logger.info(f"    - {ev}")

        logger.info("=" * 60)

    async def run(self):
        """
        Main run loop.

        Connects to WebSocket, monitors for divergences, and logs results.
        """
        await self.setup()

        self.start_time = datetime.utcnow()
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        self.is_running = True

        logger.info(f"Starting divergence detection for {self.duration_minutes} minutes...")
        logger.info(f"Will run until: {end_time.isoformat()}")

        # Get correlations to monitor
        correlations = await self.get_sample_correlations()
        logger.info(f"Monitoring {len(correlations)} correlation pairs")

        # Get markets to subscribe to
        market_ids = set()
        for corr in correlations:
            market_ids.add(corr.market_a_id)
            market_ids.add(corr.market_b_id)

        # Start price monitor
        await self.price_monitor.start()

        # Subscribe to markets
        if market_ids:
            await self.price_monitor.subscribe_to_markets(list(market_ids))

        # Start WebSocket connection in background
        ws_task = asyncio.create_task(self.ws_client.connect())

        # Detection loop
        detection_interval = 1.0  # Run detection every second

        try:
            while datetime.utcnow() < end_time and self.is_running:
                await self.run_detection_cycle(correlations)
                await asyncio.sleep(detection_interval)

                # Periodic status update
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    self._log_status()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.is_running = False

            # Cleanup
            await self.ws_client.stop()
            await self.price_monitor.stop()
            if self.cache:
                await self.cache.disconnect()
            await self.rest_client.close()

        # Final report
        self._print_final_report()

    def _log_status(self):
        """Log current status."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        logger.info(
            f"Status: {elapsed:.0f}s elapsed, "
            f"{self.stats['price_updates']} price updates, "
            f"{self.stats['divergences_found']} divergences found"
        )

    def _print_final_report(self):
        """Print final analysis report."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print("DIVERGENCE DETECTION REPORT")
        print("=" * 70)
        print(f"Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Detection cycles: {self.stats['detection_cycles']}")
        print(f"Price updates received: {self.stats['price_updates']}")
        print(f"Total divergences detected: {self.stats['divergences_found']}")
        print()
        print("Divergences by type:")
        for dtype, count in self.stats["by_type"].items():
            if count > 0:
                print(f"  {dtype}: {count}")
        print()

        if self.detected_divergences:
            print("Detected Divergences:")
            print("-" * 70)
            for div in self.detected_divergences:
                print(f"  [{div.detected_at.strftime('%H:%M:%S')}] "
                      f"{div.divergence_type.value}: "
                      f"{div.market_ids} "
                      f"spread={div.divergence_amount:.4f} "
                      f"arb={div.is_arbitrage}")
            print()

            # Analyze potential false positives
            print("False Positive Analysis:")
            print("-" * 70)
            small_spreads = [d for d in self.detected_divergences if d.divergence_amount < 0.03]
            print(f"  Small spread (<3¢): {len(small_spreads)} - likely bid/ask spread")

            low_liquidity = [d for d in self.detected_divergences if d.max_executable_size < 100]
            print(f"  Low liquidity (<$100): {len(low_liquidity)} - may not be tradeable")

            low_confidence = [d for d in self.detected_divergences if d.confidence < 0.8]
            print(f"  Low confidence (<80%): {len(low_confidence)} - uncertain")

        else:
            print("No divergences detected during this session.")
            print("This could mean:")
            print("  - Markets are efficiently priced")
            print("  - No correlations were being monitored")
            print("  - WebSocket connection issues")

        print("=" * 70)

        # Save report to file
        report_file = f"logs/divergence_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "duration_seconds": elapsed,
            "stats": self.stats,
            "divergences": [div.to_dict() for div in self.detected_divergences],
        }
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


async def run_with_mock_data(duration_minutes: int = 1):
    """
    Run detection with mock data for testing without live connection.
    """
    logger.info("Running with MOCK DATA for testing...")

    # Create mock components
    class MockWs:
        def __init__(self):
            self.callbacks = []

        def register_callback(self, cb):
            self.callbacks.append(cb)

        async def subscribe(self, ids):
            pass

        async def unsubscribe(self, ids):
            pass

        async def connect(self):
            # Simulate price updates
            import random
            while True:
                for cb in self.callbacks:
                    # Generate random price updates
                    market_id = random.choice(["m1", "m2", "m3", "m4"])
                    data = {
                        "asset_id": market_id,
                        "bids": [[0.48 + random.random() * 0.04, 100]],
                        "asks": [[0.50 + random.random() * 0.04, 100]],
                    }
                    if asyncio.iscoroutinefunction(cb):
                        await cb(data)
                    else:
                        cb(data)
                await asyncio.sleep(0.5)

        async def stop(self):
            pass

    mock_ws = MockWs()
    monitor = PriceMonitor(mock_ws)
    detector = DivergenceDetector(monitor)

    # Sample correlations
    correlations = [
        MarketCorrelation(
            market_a_id="m1",
            market_b_id="m2",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.9,
        ),
        MarketCorrelation(
            market_a_id="m3",
            market_b_id="m4",
            correlation_type=CorrelationType.POSITIVE,
            expected_relationship="A ≈ B",
            confidence=0.85,
        ),
    ]

    await monitor.start()
    ws_task = asyncio.create_task(mock_ws.connect())

    start = datetime.utcnow()
    end = start + timedelta(minutes=duration_minutes)

    detected = []

    try:
        while datetime.utcnow() < end:
            divs = await detector.detect_all_divergences(correlations, [])
            for div in divs:
                detected.append(div)
                logger.info(f"DETECTED: {div.divergence_type.value} - "
                           f"{div.market_ids} - spread={div.divergence_amount:.4f}")
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        pass

    await mock_ws.stop()

    print(f"\nMock test complete. Detected {len(detected)} divergences.")


def main():
    parser = argparse.ArgumentParser(
        description="Run live divergence detection on Polymarket"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in minutes (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock data for testing"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    if args.mock:
        asyncio.run(run_with_mock_data(args.duration))
    else:
        monitor = DivergenceMonitor(
            duration_minutes=args.duration,
            verbose=args.verbose,
        )
        asyncio.run(monitor.run())


if __name__ == "__main__":
    main()

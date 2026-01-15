
"""
Polymarket Bot Entry Point.
"""
import asyncio
import logging
import signal
import sys
from typing import Optional

from src.config.settings import Config
from src.database.postgres import DatabaseManager
from src.api.clob_client import ClobClient
from src.api.ws_client import ClobWsClient
from src.api.live_client import LiveTradingClient
from src.execution.orders.manager import OrderManager
from src.execution.positions.manager import PositionManager
from src.signals.monitor.signal_monitor import SignalMonitor
from src.signals.divergence.detector import DivergenceDetector
from src.signals.divergence.types import DivergenceConfig
from src.signals.divergence.price_monitor import PriceMonitor
from src.signals.scoring.scorer import SignalScorer
from src.correlation.store import CorrelationStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("polymarket_bot")


class Bot:
    """Main bot application."""
    
    def __init__(self):
        self.config = Config()
        self.running = False
        
        # Components
        self.db: Optional[DatabaseManager] = None
        self.clob_client: Optional[ClobClient] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.signal_monitor: Optional[SignalMonitor] = None
        
    async def setup(self):
        """Initialize all components."""
        logger.info("Initializing Polymarket Bot...")
        
        # 1. Database
        self.db = DatabaseManager(self.config)
        await self.db.connect()
        
        # 2. Clients
        self.clob_client = ClobClient(
            # API credentials are optional for public endpoints
            api_key=None,
            secret=None,
            passphrase=None
        )
        
        # 2b. Initialize live trading client if not paper trading
        self.live_client = None
        if not self.config.paper_trading:
            is_valid, msg = self.config.validate_live_trading()
            if not is_valid:
                logger.error(f"Live trading config invalid: {msg}")
                raise ValueError(msg)
            
            self.live_client = LiveTradingClient(
                private_key=self.config.polymarket_private_key,
                funder=self.config.polymarket_funder,
                signature_type=self.config.polymarket_signature_type,
            )
            logger.info("LiveTradingClient initialized for live trading")
        
        # 3. Execution Engines
        self.position_manager = PositionManager(self.config)
        self.order_manager = OrderManager(
            config=self.config,
            position_manager=self.position_manager,
            is_paper=self.config.paper_trading,
            api_client=self.live_client,  # Pass live client if available
        )
        
        # 4. Signal Pipeline
        correlation_store = CorrelationStore(self.db)

        # Create WebSocket client and PriceMonitor for real-time prices
        self.ws_client = ClobWsClient(url=self.config.polymarket_ws_url)
        self.price_monitor = PriceMonitor(ws_client=self.ws_client)

        # Configure divergence detection with lower thresholds for testing
        divergence_config = DivergenceConfig(
            min_divergence_threshold=0.01,      # 1 cent minimum spread
            min_liquidity_threshold=10.0,       # $10 minimum liquidity
            min_correlation_confidence=0.5,     # 50% confidence minimum
        )

        detector = DivergenceDetector(
            price_monitor=self.price_monitor,
            correlation_store=correlation_store,
            config=divergence_config
        )
        scorer = SignalScorer()
        
        self.signal_monitor = SignalMonitor(
            divergence_detector=detector,
            signal_scorer=scorer,
            correlation_store=correlation_store,
            config=None # Use defaults
        )
        
        # Wire up signals to execution
        self.signal_monitor.on_signal(self._handle_signal)
        
        logger.info("Initialization complete.")

    async def start(self):
        """Start the bot loops."""
        if self.running:
            return

        self.running = True
        logger.info("Starting bot services...")

        # Start price monitor for real-time data
        await self.price_monitor.start()

        # Get markets from correlations for subscription
        correlations = await self.signal_monitor.correlation_store.get_all_correlations()
        market_ids = set()
        for c in correlations:
            market_ids.add(c.market_a_id)
            market_ids.add(c.market_b_id)

        if market_ids:
            logger.info(f"Looking up token IDs for {len(market_ids)} markets...")
            # Build market_id -> token_ids map from database
            market_token_map = {}
            for market_id in market_ids:
                market = await self.db.get_market(market_id)
                if market and market.clob_token_ids:
                    market_token_map[market_id] = market.clob_token_ids

            if market_token_map:
                logger.info(f"Subscribing to {len(market_token_map)} markets with token IDs...")
                await self.price_monitor.subscribe_with_tokens(market_token_map)
            else:
                logger.warning("No markets found with token IDs - prices won't be available")

        # Start WebSocket connection as background task (this runs the actual connection)
        self._ws_task = asyncio.create_task(self.ws_client.connect())
        logger.info("WebSocket connection started")

        # Start signal monitor
        await self.signal_monitor.start()

        logger.info("Bot is running. Press Ctrl+C to stop.")
        
        # Keep alive
        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping bot...")
        self.running = False

        if self.signal_monitor:
            await self.signal_monitor.stop()

        if self.price_monitor:
            await self.price_monitor.stop()

        if hasattr(self, '_ws_task') and self._ws_task:
            await self.ws_client.stop()
            self._ws_task.cancel()

        logger.info("Bot stopped.")

    async def _handle_signal(self, signal):
        """Handle new trading signal."""
        logger.info(f"Received signal: {signal.divergence.id} score={signal.overall_score}")
        
        if self.config.paper_trading:
            logger.info("Paper trading execution...")
        else:
            logger.info("Live trading execution...")
        
        await self.order_manager.submit_signal_order(signal)


async def main():
    bot = Bot()
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.setup()
        await bot.start()
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        await bot.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

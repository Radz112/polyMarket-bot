import asyncio
import logging
from src.database.postgres import DatabaseManager
from src.correlation.store import CorrelationStore
from src.api.ws_client import ClobWsClient
from src.signals.divergence.price_monitor import PriceMonitor
from src.signals.divergence.detector import DivergenceDetector

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DivergenceMonitor")

async def main():
    logger.info("Starting Divergence Monitor...")
    
    # 1. Setup Data Layers
    db = DatabaseManager()
    await db.connect()
    store = CorrelationStore(db)
    
    # 2. Load Correlations & Rules
    logger.info("Loading correlations and rules...")
    # Mocking for script demonstration if DB is empty, or load real data
    # Assuming DB has data from Step 2.5 integration
    try:
        correlations = await store.get_all_correlations()
        # rules = await store.get_logical_rules() # Not implemented in store yet?
        rules = [] # Placeholder
        logger.info(f"Loaded {len(correlations)} correlations.")
    except Exception as e:
        logger.warning(f"Failed to load data from DB: {e}. Using MOCK data for verification.")
        # MOCK DATA FALLBACK
        from src.models import MarketCorrelation, CorrelationType
        correlations = [
            MarketCorrelation(
                market_a_id="m1-btc-90k",
                market_b_id="m2-btc-100k",
                correlation_type=CorrelationType.POSITIVE,
                expected_relationship="Threshold",
                confidence=1.0,
                metadata={"detection_methods": ["logical"], "lead_lag_seconds": 0}
            ),
             MarketCorrelation(
                market_a_id="m3-leader",
                market_b_id="m4-follower",
                correlation_type=CorrelationType.POSITIVE,
                expected_relationship="LeadLag",
                confidence=0.95,
                metadata={"lead_lag_seconds": 60, "leader_market_id": "m3-leader"}
            )
        ]
        from src.correlation.logical.rules import LogicalRule, LogicalRuleType
        rules = [
             LogicalRule(
                rule_type=LogicalRuleType.THRESHOLD_ORDERING,
                market_ids=["m1-btc-90k", "m2-btc-100k"],
                constraint_desc="P(90k) >= P(100k)",
                metadata={"lower_strike_market": "m1-btc-90k", "higher_strike_market": "m2-btc-100k"}
            )
        ]
        logger.info(f"Loaded {len(correlations)} MOCK correlations.")

    # 3. Setup WebSocket & Monitor
    ws_client = ClobWsClient()
    monitor = PriceMonitor(ws_client)
    
    # Subscribe to all relevant markets
    market_ids = set()
    for c in correlations:
        market_ids.add(c.market_a_id)
        market_ids.add(c.market_b_id)
    
    if not market_ids:
        logger.warning("No markets to monitor. Exiting.")
        await db.disconnect()
        return

    # Hook up price updates (WS client -> Monitor)
    # The monitor doesn't auto-hook in our current impl, we need a bridge.
    # We register a callback on ws_client that feeds monitor.
    def ws_callback(msg):
        # Parse MSG and update monitor (Simplified)
        # Real msg parsing needed here depending on Polymarket format
        pass 
    
    ws_client.register_callback(ws_callback)
    
    # 4. Start Detector Loop
    detector = DivergenceDetector(monitor)
    
    logger.info(f"Monitoring {len(market_ids)} markets via WebSocket...")
    # asyncio.create_task(ws_client.connect()) # Start WS background
    
    # INJECT MOCK PRICES FOR VERIFICATION
    logger.info("Injecting mock prices to simulate opportunities...")
    # 1. Threshold Violation (90k cheaper than 100k)
    monitor.on_price_update("m1-btc-90k", 0.50)
    monitor.on_price_update("m2-btc-100k", 0.60) # EXPENSIVE!
    
    # 2. Lead Lag (Leader moved, Follower stagnant)
    # Need history for get_price_change(lag=60)
    monitor._price_history["m3-leader"] = [(1000, 0.50), (1060, 0.60)] # Moved up
    monitor._market_prices["m3-leader"] = 0.60
    
    monitor._price_history["m4-follower"] = [(1000, 0.50), (1060, 0.50)] # Stagnant
    monitor._market_prices["m4-follower"] = 0.50
    
    # Hack time.time for monitor history check
    import time
    original_time = time.time
    time.time = lambda: 1060 # Mock current time

    # Simulation Loop
    try:
        while True:
            # In real app, this runs continuously. 
            # Here we just run detection once to show logic.
            
            divergences = detector.detect_all(correlations, rules)
            
            if divergences:
                logger.info(f"Detected {len(divergences)} divergences!")
                for div in divergences:
                     # Check if it's the right type
                    logger.info(f" - {div.divergence_type.name} | {div.market_ids} | Profit: {div.profit_potential:.2f}")
            else:
                 logger.info("No divergences detected in this cycle.")

            await asyncio.sleep(1)
            # Break for script termination
            break 
    except KeyboardInterrupt:
        pass
    finally:
        time.time = original_time # Restore
        await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

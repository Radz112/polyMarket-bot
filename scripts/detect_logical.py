import logging
import asyncio
from src.models import Market
from src.correlation.logical.detector import LogicalRuleDetector
from src.correlation.logical.violations import ViolationDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LogicalVerifier")

def main():
    logger.info("Starting Logical Correlation Verification...")
    
    # 1. Create Mock Markets (Broken Chain)
    m1 = Market(condition_id="m1", slug="btc-90k", question="Will Bitcoin be above $90,000 on Dec 31?", entities={})
    m2 = Market(condition_id="m2", slug="btc-100k", question="Will Bitcoin be above $100,000 on Dec 31?", entities={})
    m3 = Market(condition_id="m3", slug="btc-110k", question="Will Bitcoin be above $110,000 on Dec 31?", entities={})
    
    markets = [m1, m2, m3]
    logger.info(f"Created {len(markets)} mock markets for BTC threshold chain.")
    
    # 2. Detect Rules
    detector = LogicalRuleDetector()
    rules = detector.detect_rules(markets)
    logger.info(f"Detected {len(rules)} logical rules.")
    
    for r in rules:
        logger.info(f"Rule: {r.constraint_desc}")
        
    # 3. Simulate Prices (Inverted to create arbitrage)
    # >90k: 60c (Correct ordering vs 100k)
    # >100k: 40c
    # >110k: 45c (INVERTED! >110k is more expensive than >100k)
    
    prices = {
        "m1": 0.60,
        "m2": 0.40,
        "m3": 0.45 
    }
    logger.info(f"Simulating Prices: {prices}")
    
    # 4. Check Violations
    v_detector = ViolationDetector()
    violations = []
    for r in rules:
        v = v_detector.check_rule(r, prices)
        if v:
            violations.append(v)
            
    logger.info(f"Found {len(violations)} rule violations.")
    
    print("\n=== Violation Report ===")
    for v in violations:
        print(f"Rule: {v.rule.constraint_desc}")
        print(f"Details: {v.details}")
        print(f"Profit Opportunity: {v.profit_opportunity:.2f}")
        for trade in v.suggested_trades:
            print(f"  -> Trade: {trade[1]} on {trade[0]}")
            
    # Verification
    # Expected violation between m2 and m3
    # m3 (110k) > m2 (100k)
    assert len(violations) >= 1
    found = any(v.rule.metadata["lower_strike_market"] == "m2" and v.rule.metadata["higher_strike_market"] == "m3" for v in violations)
    if found:
        logger.info("SUCCESS: Correctly detected price inversion between 100k and 110k strikes.")
    else:
        logger.error("FAILURE: Did not detect the expected violation.")
        exit(1)

if __name__ == "__main__":
    main()

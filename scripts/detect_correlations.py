import asyncio
import logging
import time
from src.models.market import Market
from src.correlation.similarity.detector import StringSimilarityDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CorrelationVerifier")

def create_mock_markets():
    # Similar Pair 1: Politics
    m1 = Market(condition_id="1", slug="trump-2024", question="Will Donald Trump win the 2024 US Presidential Election?", category="Politics", subcategory="US Presidential", entities={"people": ["Donald Trump"], "custom": {"years": ["2024"]}})
    m2 = Market(condition_id="2", slug="trump-pres", question="Will Trump be elected President in 2024?", category="Politics", subcategory="US Presidential", entities={"people": ["Trump"], "custom": {"years": ["2024"]}})
    
    # Similar Pair 2: Crypto
    m3 = Market(condition_id="3", slug="btc-100k", question="Bitcoin above $100k by EOY 2024?", category="Crypto", subcategory="Prices", entities={"custom": {"prices": ["$100k"], "years": ["2024"]}})
    m4 = Market(condition_id="4", slug="eth-10k", question="Ethereum above $10k by EOY 2024?", category="Crypto", subcategory="Prices", entities={"custom": {"prices": ["$10k"], "years": ["2024"]}}) 
    # m3/m4 should have some similarity (category, date) but not extremely high
    
    # Similar Pair 3: Overlap
    m5 = Market(condition_id="5", slug="btc-hit-100k", question="Will BTC hit 100,000 in 2024?", category="Crypto", subcategory="Prices", entities={"custom": {"prices": ["100000"], "years": ["2024"]}})
    
    # Random
    m6 = Market(condition_id="6", slug="superbowl", question="Will Chiefs win Super Bowl?", category="Sports", subcategory="NFL", entities={"orgs": ["Chiefs"]})
    
    return [m1, m2, m3, m4, m5, m6]

async def main():
    logger.info("Starting Correlation Detection Verification")
    
    markets = create_mock_markets()
    logger.info(f"Created {len(markets)} mock markets.")
    
    detector = StringSimilarityDetector(threshold=0.4) # Low threshold to see what happens
    
    start_time = time.time()
    results = detector.find_similar_markets(markets)
    end_time = time.time()
    
    logger.info(f"Detection took {end_time - start_time:.4f}s")
    logger.info(f"Found {len(results)} correlated pairs.")
    
    print("\n=== Top Correlated Pairs ===")
    for res in results:
        m_a = next(m for m in markets if m.id == res.market_a_id)
        m_b = next(m for m in markets if m.id == res.market_b_id)
        print(f"[{res.overall_score:.2f}] {m_a.slug} <-> {m_b.slug}")
        print(f"    Q1: {m_a.question}")
        print(f"    Q2: {m_b.question}")
        print(f"    Breakdown: {res.breakdown}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())

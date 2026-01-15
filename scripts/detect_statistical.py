import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.correlation.statistical.timeseries import PriceAnalyzer
from src.correlation.statistical.leadlag import LeadLagDetector
from src.models import PriceSnapshot

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StatisticalVerifier")

def generate_correlated_walks(n=500, correlation=0.9):
    """Generate two correlated random walks."""
    # X ~ N(0,1)
    # Y = rho * X + sqrt(1 - rho^2) * Z, where Z ~ N(0,1)
    # Then cumsum them
    dt = 0.01
    x = np.random.normal(0, 1, n)
    z = np.random.normal(0, 1, n)
    y = correlation * x + np.sqrt(1 - correlation**2) * z
    
    price_a = 0.5 + np.cumsum(x * dt)
    price_b = 0.5 + np.cumsum(y * dt)
    
    # Clamp to 0-1
    price_a = np.clip(price_a, 0.01, 0.99)
    price_b = np.clip(price_b, 0.01, 0.99)
    
    return price_a, price_b

def main():
    logger.info("Starting Statistical Correlation Verification (Simulation)...")
    
    analyzer = PriceAnalyzer()
    lead_lag = LeadLagDetector()
    
    # 1. Generate Data
    logger.info("Generating synthetic correlated prices...")
    pa, pb = generate_correlated_walks(n=1000, correlation=0.95)
    
    # Create Snapshots
    start_time = datetime.utcnow()
    snapshots_a = []
    snapshots_b = []
    
    for i in range(len(pa)):
        t = start_time + timedelta(minutes=i)
        snapshots_a.append(PriceSnapshot(market_id="A", timestamp=t, yes_price=pa[i], no_price=1-pa[i], yes_volume=0, no_volume=0))
        # Add some jitter/gaps to B? No, let's test clean first.
        # Let's lag B by 5 mins to test lead/lag
        # B[t] corresponds to A[t-5]
        if i >= 5:
            t_lag = start_time + timedelta(minutes=i) # same clock time
            # price is from 5 steps ago
            snapshots_b.append(PriceSnapshot(market_id="B", timestamp=t_lag, yes_price=pa[i-5], no_price=1-pa[i-5], yes_volume=0, no_volume=0))

    # 2. Analyze
    ts_a = analyzer.to_series(snapshots_a)
    ts_b = analyzer.to_series(snapshots_b)
    
    aligned_a, aligned_b = analyzer.align_time_series(ts_a, ts_b)
    
    logger.info(f"Aligned {len(aligned_a)} data points.")
    
    # 3. Lead Lag
    logger.info("Calculating Lead/Lag details...")
    res = lead_lag.find_lead_lag(aligned_a, aligned_b, max_lag_seconds=600, interval_seconds=60)
    
    print("\n=== Verification Results ===")
    print(f"Leader: {res.leader}")
    print(f"Lag: {res.lag_seconds}s")
    print(f"Confidence: {res.confidence:.4f}")
    
    assert res.leader == "market_a"
    # We injected 5 min lag = 300s
    assert res.lag_seconds == 300
    
    logger.info("SUCCESS: Detected correct lead/lag relationship.")

    # 4. Correlation of returns
    rets_a = aligned_a.pct_change().dropna()
    rets_b = aligned_b.pct_change().dropna()
    # At lag 0, correlation might be low because of the lag
    immed_corr = rets_a.corr(rets_b)
    print(f"Immediate Correlation (should be low due to lag): {immed_corr:.4f}")
    
    # At optimal lag
    best_corr = float(res.confidence)
    print(f"Lagged Correlation (should be high): {best_corr:.4f}")
    
if __name__ == "__main__":
    main()

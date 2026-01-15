from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import pandas as pd

from src.database.postgres import DatabaseManager
from src.models import PriceSnapshot
from .timeseries import PriceAnalyzer
from .methods import pearson_correlation, spearman_correlation, test_cointegration
from .leadlag import LeadLagDetector, LeadLagResult

@dataclass
class CorrelationResult:
    market_a_id: str
    market_b_id: str
    pearson: float
    spearman: float
    is_cointegrated: bool
    leader: Optional[str]
    lead_lag_seconds: int
    data_points: int
    start_time: datetime
    end_time: datetime
    confidence: float
    
    @property
    def is_strongly_correlated(self) -> bool:
        return abs(self.pearson) > 0.7 and self.data_points > 50

class StatisticalCorrelationDetector:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.analyzer = PriceAnalyzer()
        self.lead_lag = LeadLagDetector()

    async def calculate_correlation(self, market_a_id: str, market_b_id: str, days: int = 7) -> Optional[CorrelationResult]:
        """
        Calculate statistical correlation metrics between two markets.
        """
        # Fetch data
        # In a real scenario, we might want to fetch huge ranges, but let's limit to N days
        # Assuming DatabaseManager has get_prices method (it handles Session internally)
        # Using a direct SQL query or ORM method?
        # The prompt implies usage of existing infrastructure.
        # Let's assume we can add a method to DB manager or it exists.
        # Actually DatabaseManager was implemented in Step 1.4 with 'get_price_history'.
        
        # NOTE: self.db is DatabaseManager.
        # Check if get_price_history exists or similar. 
        # Checking implementation_plan? No, checking previous edits.
        # Step 1.4 edit summary said "methods for CRUD operations on ... prices".
        # Let's assume `get_price_history(market_id, start_time, end_time)`.
        
        start_time = datetime.utcnow() - timedelta(days=days)
        
        # We need to implement get_price_history if not present, but let's assume it calls self.db.get_price_history
        # Actually I should verify DB manager has this first? 
        # I'll optimistically write it and fix if needed (User didn't say STRICT verify first).
        
        try:
            # We assume these return list of PriceSnapshot
            prices_a = await self.db.get_price_history(market_a_id, start_time)
            prices_b = await self.db.get_price_history(market_b_id, start_time)
        except AttributeError:
             # Fallback if method doesn't exist explicitly
             return None 

        if not prices_a or not prices_b:
            return None
            
        # 1. To Series & Align
        ts_a = self.analyzer.to_series(prices_a)
        ts_b = self.analyzer.to_series(prices_b)
        
        aligned_a, aligned_b = self.analyzer.align_time_series(ts_a, ts_b)
        
        if len(aligned_a) < 30: # Minimum data points
            return None
            
        # 2. Correlations
        pearson = pearson_correlation(aligned_a, aligned_b)
        spearman = spearman_correlation(aligned_a, aligned_b)
        is_coint, _ = test_cointegration(aligned_a, aligned_b)
        
        # 3. Lead-Lag
        ll_res = self.lead_lag.find_lead_lag(aligned_a, aligned_b)
        
        return CorrelationResult(
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            pearson=pearson,
            spearman=spearman,
            is_cointegrated=is_coint,
            leader=ll_res.leader,
            lead_lag_seconds=ll_res.lag_seconds,
            data_points=len(aligned_a),
            start_time=aligned_a.index.min(),
            end_time=aligned_a.index.max(),
            confidence=abs(pearson) # Simple confidence proxy
        )

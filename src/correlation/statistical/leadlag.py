import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class LeadLagResult:
    leader: Optional[str] # "market_a", "market_b", or None
    lag_seconds: int
    confidence: float
    cross_correlation: Dict[int, float]

class LeadLagDetector:
    def __init__(self):
        pass

    def find_lead_lag(self, series_a: pd.Series, series_b: pd.Series, max_lag_seconds: int = 300, interval_seconds: int = 60) -> LeadLagResult:
        """
        Determine if one market leads the other using cross-correlation.
        series_a and series_b should be aligned Price Series (not returns), or Returns?
        Returns are usually better for lead-lag to avoid trend bias.
        """
        # Calculate returns if not already stationary
        # Ideally passed in series are stationary returns
        # But if passed prices, we diff them.
        
        # Check stationarity heuristic: if mean/std drift, use diff
        # Simple approach: always use pct_change for lead-lag
        returns_a = series_a.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        returns_b = series_b.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

        lags = range(-max_lag_seconds // interval_seconds, (max_lag_seconds // interval_seconds) + 1)
        corrs = {}
        
        for lag in lags:
            if lag == 0:
                c = returns_a.corr(returns_b)
            elif lag < 0:
                # A shifted back relative to B?
                # shift(-k) moves data UP (future values).
                # shift(k) moves data DOWN (past values).
                
                # If lag is negative, we shift B?
                # Let's standardize:
                # corr(A_t, B_{t+lag})
                shifted_b = returns_b.shift(-lag) # shift(positive)
                c = returns_a.corr(shifted_b)
            else:
                shifted_b = returns_b.shift(-lag) # shift(negative)
                c = returns_a.corr(shifted_b)
            
            if not np.isnan(c):
                corrs[lag * interval_seconds] = c
                
        # Find max correlation
        if not corrs:
            return LeadLagResult(None, 0, 0.0, {})
            
        best_lag = max(corrs, key=corrs.get)
        max_corr = corrs[best_lag]
        
        # If max correlation is at 0, no lead/lag or instant
        # If best_lag > 0 => A correlates with Future B => A leads B?
        # corr(A_t, B_{t+k}) is high => A_t predicts B_{t+k} => A leads.
        
        leader = None
        if abs(best_lag) > 0 and max_corr > 0.1: # Threshold noise
            if best_lag > 0:
                leader = "market_a"
            else:
                leader = "market_b"
                
        return LeadLagResult(
            leader=leader,
            lag_seconds=abs(best_lag),
            confidence=max_corr,
            cross_correlation=corrs
        )

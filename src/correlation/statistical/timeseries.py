import pandas as pd
from typing import List, Tuple
from datetime import datetime
from src.models import PriceSnapshot

class PriceAnalyzer:
    def __init__(self):
        pass

    def to_series(self, snapshots: List[PriceSnapshot], frequency: str = "1min") -> pd.Series:
        """
        Convert list of PriceSnapshots to a Resampled Pandas Series.
        Snapshots are expected to be sorted by time (or we sort them).
        """
        if not snapshots:
            return pd.Series(dtype=float)
            
        data = {
            "timestamp": [s.timestamp for s in snapshots],
            "price": [s.yes_price for s in snapshots]
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Resample and forward fill (last known price is valid until new one)
        # Then backward fill for any leading gaps if necessary, or drop them?
        # Usually for correlation we want overlapping valid data.
        # ffill() propagates last valid observation forward.
        series = df["price"].resample(frequency).last().ffill()
        
        return series

    def align_time_series(self, series_a: pd.Series, series_b: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Align two series to the same index (intersection of timestamps).
        """
        # Outer join to capture full range, then ffill?
        # Correlation usually requires values at same timestamps.
        # If we align by intersection ('inner'), we only compare when both existed.
        # But if one market is illiquid, we might want to ffill its price to matched timestamps.
        
        # Strategy: Combine, forward fill both to fill gaps, then dropna
        df = pd.concat([series_a, series_b], axis=1, keys=['a', 'b'])
        df = df.ffill().dropna()
        
        return df['a'], df['b']

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate percentage change."""
        return prices.pct_change().fillna(0)

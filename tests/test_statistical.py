import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.correlation.statistical.timeseries import PriceAnalyzer
from src.correlation.statistical.methods import pearson_correlation, rolling_correlation
from src.correlation.statistical.leadlag import LeadLagDetector
from src.models import PriceSnapshot

@pytest.fixture
def analyzer():
    return PriceAnalyzer()

@pytest.fixture
def leadlag():
    return LeadLagDetector()

def test_to_series_resampling(analyzer):
    now = datetime(2024, 1, 1, 12, 0)
    snapshots = [
        PriceSnapshot(market_id="1", timestamp=now, yes_price=0.5, no_price=0.5, yes_volume=0, no_volume=0),
        PriceSnapshot(market_id="1", timestamp=now + timedelta(minutes=2), yes_price=0.6, no_price=0.4, yes_volume=0, no_volume=0),
         # Gap of 2 mins
        PriceSnapshot(market_id="1", timestamp=now + timedelta(minutes=5), yes_price=0.7, no_price=0.3, yes_volume=0, no_volume=0),
    ]
    
    # 1min resampling
    series = analyzer.to_series(snapshots, frequency="1min")
    
    # Values at:
    # 12:00 -> 0.5 (actual)
    # 12:01 -> 0.5 (ffill)
    # 12:02 -> 0.6 (actual)
    # 12:03 -> 0.6 (ffill)
    # 12:04 -> 0.6 (ffill)
    # 12:05 -> 0.7 (actual)
    assert len(series) == 6
    assert series.iloc[0] == 0.5
    assert series.iloc[1] == 0.5
    assert series.iloc[2] == 0.6

def test_pearson_correlation():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([2, 4, 6, 8, 10]) # Perfect linear
    score = pearson_correlation(s1, s2)
    assert score > 0.99

    s3 = pd.Series([5, 4, 3, 2, 1]) # Perfect inverse
    score_inv = pearson_correlation(s1, s3)
    assert score_inv < -0.99

def test_lead_lag(leadlag):
    # A leads B by 1 step
    idx = pd.date_range("2024-01-01", periods=1000, freq="1min")
    data_a = np.random.randn(1000).cumsum() # Random walk
    series_a = pd.Series(data_a, index=idx)
    
    # Series B is same but shifted 2 mins later
    data_b = series_a.shift(2).fillna(0)
    series_b = pd.Series(data_b, index=idx)
    
    # We pass PRICES, internally it does pct_change
    # However, random walk diff is stationary noise.
    # To detect lead lag from prices, the logic MUST do pct_change
    res = leadlag.find_lead_lag(series_a, series_b, interval_seconds=60)
    
    # A leads B
    assert res.leader == "market_a"
    assert res.lag_seconds == 120 # 2 mins
    assert res.confidence > 0.5

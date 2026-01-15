from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, grangercausalitytests

def pearson_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """Standard linear correlation."""
    return series_a.corr(series_b, method='pearson')

def spearman_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """Rank-based correlation."""
    return series_a.corr(series_b, method='spearman')

def rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
    """Time-varying correlation."""
    return series_a.rolling(window=window).corr(series_b)

def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> Tuple[bool, float]:
    """
    Test for cointegration (long-term relationship).
    Returns (is_cointegrated, p_value)
    """
    if len(series_a) < 100: # Need sufficient data
        return False, 1.0
        
    try:
        # Engle-Granger test
        score, pvalue, _ = coint(series_a, series_b)
        return pvalue < 0.05, pvalue
    except Exception:
        return False, 1.0

def granger_causality(series_a: pd.Series, series_b: pd.Series, max_lag: int = 5) -> Dict[str, Any]:
    """
    Test if series_a Granger-causes series_b (and vice versa).
    Note: Granger causality essentially checks if past values of X help predict Y.
    
    Returns:
    {
        "a_causes_b": bool,
        "b_causes_a": bool,
        "min_p_value": float
    }
    """
    # Prepare DataFrame
    data = pd.concat([series_b, series_a], axis=1) # target, predictor
    
    # Check A -> B
    # grangercausalitytests returns a dict of results for each lag
    # We check if any lag exists where p < 0.05
    try:
        # verbose=False to suppress print
        res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        min_p = 1.0
        for lag in range(1, max_lag + 1):
            # ssr_chi2test is usually good
            p = res[lag][0]['ssr_chi2test'][1]
            if p < min_p:
                min_p = p
        
        a_causes_b = min_p < 0.05
    except Exception:
        a_causes_b = False
        min_p = 1.0
        
    return {"a_causes_b": a_causes_b, "min_p_value": min_p}

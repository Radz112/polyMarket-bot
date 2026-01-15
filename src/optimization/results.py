"""
Optimization results and analysis models.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.backtesting.results import BacktestResults


@dataclass
class OptimizationResult:
    """Result of a single optimization run (one parameter set)."""
    params: Dict[str, Any]
    objective_value: float
    backtest_result: BacktestResults


@dataclass
class OverfittingAnalysis:
    """Analysis of potential overfitting."""
    in_sample_performance: float
    out_sample_performance: float
    degradation_pct: float
    is_overfit: bool


@dataclass
class OptimizationResults:
    """Collection of results from an optimization session."""
    method: str
    objective: str
    results: List[OptimizationResult]
    best_params: Dict[str, Any]
    
    # Analysis
    convergence_curve: Optional[List[float]] = None
    parameter_importance: Optional[Dict[str, float]] = None
    
    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """Get top N results."""
        return self.results[:n]
    
    def get_parameter_sensitivity(self, param_name: str) -> pd.DataFrame:
        """
        Analyze how objective changes with parameter.
        Returns DataFrame with param values and corresponding objectives.
        """
        data = [(r.params[param_name], r.objective_value) for r in self.results]
        df = pd.DataFrame(data, columns=[param_name, self.objective])
        return df.groupby(param_name)[self.objective].agg(['mean', 'std', 'count'])


class OptimizationAnalyzer:
    """Analyzes optimization results."""
    
    def __init__(self, results: OptimizationResults):
        self.results = results
    
    def calculate_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate relative importance of each parameter.
        Uses variance-based sensitivity analysis.
        """
        if not self.results.results:
            return {}
            
        importance = {}
        sample_params = self.results.results[0].params
        
        for param_name in sample_params.keys():
            try:
                sensitivity = self.results.get_parameter_sensitivity(param_name)
                # Use standard deviation of objective across parameter values as proxy for importance
                # If changing the parameter causes high variance in result, it's important
                if len(sensitivity) > 1:
                    importance[param_name] = sensitivity['mean'].std()
                else:
                    importance[param_name] = 0.0
            except Exception:
                importance[param_name] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            return {k: v/total for k, v in importance.items()}
        return importance
    
    def detect_overfitting(
        self,
        in_sample_results: OptimizationResults,
        out_sample_results: OptimizationResults
    ) -> OverfittingAnalysis:
        """
        Compare in-sample vs out-of-sample performance.
        Large degradation indicates overfitting.
        """
        if not in_sample_results.results or not out_sample_results.results:
            return OverfittingAnalysis(0, 0, 0, False)

        in_sample_best = in_sample_results.results[0].objective_value
        best_params = in_sample_results.best_params
        
        # Find same params in out-sample
        out_sample_value = None
        for r in out_sample_results.results:
            # Need strict equality for params dict
            if r.params == best_params:
                out_sample_value = r.objective_value
                break
        
        if out_sample_value is None:
            # If exact params not found in out-sample (e.g. random search), take best
            # This is a fallback but less accurate for strict overfitting check
            out_sample_value = out_sample_results.results[0].objective_value
            
        degradation = 0.0
        if in_sample_best != 0:
            degradation = (in_sample_best - out_sample_value) / in_sample_best
        
        return OverfittingAnalysis(
            in_sample_performance=in_sample_best,
            out_sample_performance=out_sample_value,
            degradation_pct=degradation * 100,
            is_overfit=degradation > 0.3  # >30% degradation
        )

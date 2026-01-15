"""
Grid search optimization method.
"""
import logging
from typing import List, Any
from copy import deepcopy

from src.backtesting.config import BacktestConfig
from src.optimization.parameters import ParameterSpace
from src.optimization.results import OptimizationResult, OptimizationResults

logger = logging.getLogger(__name__)


class GridSearch:
    """Exhaustive grid search optimizer."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    async def optimize(
        self,
        backtest_config: BacktestConfig
    ) -> OptimizationResults:
        """Run grid search."""
        combinations = self.optimizer.parameter_space.get_grid_combinations()
        logger.info(f"Grid search: {len(combinations)} combinations")
        
        results = []
        for i, params in enumerate(combinations):
            logger.info(f"Grid search step {i+1}/{len(combinations)}")
            
            # Create config copy with new params
            # Note: Strategy params are usually passed to Strategy, not Config directly
            # But Backtester/Strategy needs a way to receive them.
            # We assume here that we apply them to strategy or config as handled by optimizer.apply_params
            
            backtest_result = await self.optimizer.run_backtest(backtest_config, params)
            objective_value = self.optimizer.get_objective_value(backtest_result)
            
            results.append(OptimizationResult(
                params=params,
                objective_value=objective_value,
                backtest_result=backtest_result
            ))
        
        sorted_results = sorted(results, key=lambda r: r.objective_value, reverse=True)
        
        return OptimizationResults(
            method="grid",
            objective=self.optimizer.objective,
            results=sorted_results,
            best_params=sorted_results[0].params if sorted_results else {}
        )

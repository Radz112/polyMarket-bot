"""
Random search optimization method.
"""
import logging
from typing import List, Any
from src.backtesting.config import BacktestConfig
from src.optimization.results import OptimizationResult, OptimizationResults

logger = logging.getLogger(__name__)


class RandomSearch:
    """Random search optimizer."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    async def optimize(
        self,
        backtest_config: BacktestConfig,
        num_iterations: int = 100
    ) -> OptimizationResults:
        """Run random search."""
        logger.info(f"Random search: {num_iterations} iterations")
        
        results = []
        for i in range(num_iterations):
            logger.info(f"Random search iteration {i+1}/{num_iterations}")
            
            params = self.optimizer.parameter_space.sample_random()
            
            backtest_result = await self.optimizer.run_backtest(backtest_config, params)
            objective_value = self.optimizer.get_objective_value(backtest_result)
            
            results.append(OptimizationResult(
                params=params,
                objective_value=objective_value,
                backtest_result=backtest_result
            ))
        
        sorted_results = sorted(results, key=lambda r: r.objective_value, reverse=True)
        
        return OptimizationResults(
            method="random",
            objective=self.optimizer.objective,
            results=sorted_results,
            best_params=sorted_results[0].params if sorted_results else {}
        )

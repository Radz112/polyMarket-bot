"""
Parallel optimization support.
"""
import asyncio
import logging
from typing import List

from src.backtesting.config import BacktestConfig
from src.optimization.optimizer import Optimizer
from src.optimization.results import OptimizationResult, OptimizationResults

logger = logging.getLogger(__name__)


class ParallelOptimizer:
    """Wrapper to run optimization tasks in parallel."""
    
    def __init__(self, optimizer: Optimizer, num_workers: int = 4):
        self.optimizer = optimizer
        self.num_workers = num_workers
        
    async def parallel_grid_search(
        self,
        backtest_config: BacktestConfig
    ) -> OptimizationResults:
        """Run grid search in parallel."""
        combinations = self.optimizer.parameter_space.get_grid_combinations()
        logger.info(f"Parallel Grid Search: {len(combinations)} combos, {self.num_workers} workers")
        
        # Divide work into chunks
        chunk_size = max(1, len(combinations) // self.num_workers)
        chunks = [
            combinations[i:i + chunk_size]
            for i in range(0, len(combinations), chunk_size)
        ]
        
        tasks = []
        for chunk in chunks:
            tasks.append(self._process_chunk(chunk, backtest_config))
            
        chunk_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        all_results = []
        for res_list in chunk_results:
            all_results.extend(res_list)
            
        sorted_results = sorted(all_results, key=lambda r: r.objective_value, reverse=True)
        
        return OptimizationResults(
            method="parallel_grid",
            objective=self.optimizer.objective,
            results=sorted_results,
            best_params=sorted_results[0].params if sorted_results else {}
        )
        
    async def _process_chunk(
        self,
        combinations: List[dict],
        backtest_config: BacktestConfig
    ) -> List[OptimizationResult]:
        """Process a chunk of combinations."""
        results = []
        for params in combinations:
            backtest_result = await self.optimizer.run_backtest(backtest_config, params)
            objective_value = self.optimizer.get_objective_value(backtest_result)
            
            results.append(OptimizationResult(
                params=params,
                objective_value=objective_value,
                backtest_result=backtest_result
            ))
        return results

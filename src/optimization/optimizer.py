"""
Main optimizer engine.
"""
import logging
import copy
from typing import Dict, Any, Optional

from src.backtesting.config import BacktestConfig
from src.backtesting.runner import BacktestRunner, Strategy, BacktestResults
from src.optimization.parameters import ParameterSpace
from src.optimization.results import OptimizationResult, OptimizationResults
from src.optimization.methods.grid import GridSearch
from src.optimization.methods.random import RandomSearch
from src.optimization.methods.bayesian import BayesianOptimization
from src.optimization.methods.genetic import GeneticAlgorithm

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Main strategy optimizer.
    Delegates to specific optimization methods (grid, random, etc).
    """
    
    def __init__(
        self,
        backtest_runner: BacktestRunner,
        strategy: Strategy,
        parameter_space: ParameterSpace,
        objective: str = "sharpe_ratio"
    ):
        self.backtest_runner = backtest_runner
        self.strategy = strategy
        self.parameter_space = parameter_space
        self.objective = objective
        
        # Initialize methods
        self.grid_search = GridSearch(self)
        self.random_search = RandomSearch(self)
        self.bayesian = BayesianOptimization(self)
        self.genetic = GeneticAlgorithm(self)
    
    async def optimize(
        self,
        method: str,
        backtest_config: BacktestConfig,
        **kwargs
    ) -> OptimizationResults:
        """
        Run optimization using specified method.
        
        Args:
            method: "grid", "random", "bayesian", "genetic"
            backtest_config: Base backtest configuration
            **kwargs: Method-specific arguments (e.g. num_iterations)
        """
        logger.info(f"Starting {method} optimization for {self.objective}")
        
        if method == "grid":
            return await self.grid_search.optimize(backtest_config)
        
        elif method == "random":
            num_iterations = kwargs.get("num_iterations", 20)
            return await self.random_search.optimize(backtest_config, num_iterations=num_iterations)
        
        elif method == "bayesian":
            num_iterations = kwargs.get("num_iterations", 50)
            num_initial = kwargs.get("num_initial", 10)
            return await self.bayesian.optimize(backtest_config, num_iterations=num_iterations, num_initial=num_initial)
        
        elif method == "genetic":
            population_size = kwargs.get("population_size", 20)
            generations = kwargs.get("generations", 10)
            mutation_rate = kwargs.get("mutation_rate", 0.1)
            return await self.genetic.optimize(
                backtest_config, 
                population_size=population_size, 
                generations=generations,
                mutation_rate=mutation_rate
            )
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    async def run_backtest(
        self,
        base_config: BacktestConfig,
        params: Dict[str, Any]
    ) -> BacktestResults:
        """
        Run a single backtest with given parameters.
        Applies parameters to strategy/config as needed.
        """
        # Create config copy
        config = copy.deepcopy(base_config)
        
        # Create strategy copy to avoid mutating original
        # Note: Optimization assumes strategy can be configured via attributes
        # or that we pass params in a way the strategy understands.
        # Since 'Strategy' is a Protocol in backtester.py, we need a way to set params.
        
        # We assume for now that if `strategy` has an attribute matching the param name, we set it.
        # And if `config` has it, we set it there.
        
        strategy_copy = copy.deepcopy(self.strategy)
        
        for name, value in params.items():
            # Try setting on config first
            if hasattr(config, name):
                setattr(config, name, value)
            
            # Try setting on strategy
            if hasattr(strategy_copy, name):
                setattr(strategy_copy, name, value)
                
            # If strategy has a generic 'configure' or 'params' dict
            if hasattr(strategy_copy, 'params') and isinstance(strategy_copy.params, dict):
                strategy_copy.params[name] = value

        return await self.backtest_runner.run_single(config, strategy_copy)

    def get_objective_value(self, result: BacktestResults) -> float:
        """Extract objective metric from backtest results."""
        if self.objective == "sharpe_ratio":
            return result.sharpe_ratio
        elif self.objective == "total_return":
            return result.total_return
        elif self.objective == "total_return_pct":
            return result.total_return_pct
        elif self.objective == "profit_factor":
            return result.profit_factor
        elif self.objective == "win_rate":
            return result.win_rate
        elif self.objective == "calmar_ratio":
            return result.calmar_ratio
        elif self.objective == "sortino_ratio":
            return result.sortino_ratio
        else:
            logger.warning(f"Unknown objective {self.objective}, returning 0")
            return 0.0

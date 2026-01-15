"""
Bayesian optimization method using Gaussian Processes.
"""
import logging
import numpy as np
from typing import List, Any, Dict

from src.backtesting.config import BacktestConfig
from src.optimization.results import OptimizationResult, OptimizationResults

logger = logging.getLogger(__name__)

# Conditional import for sklearn
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Bayesian optimization will fail if used.")


class BayesianOptimization:
    """
    Bayesian optimization using Gaussian Processes.
    Efficiently searches parameter space by modeling objective function.
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    async def optimize(
        self,
        backtest_config: BacktestConfig,
        num_iterations: int = 50,
        num_initial: int = 10
    ) -> OptimizationResults:
        """Run Bayesian optimization."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian optimization")
            
        logger.info(f"Bayesian optimization: {num_initial} initial + {num_iterations-num_initial} optimization steps")
        
        X = []  # Parameter arrays
        y = []  # Objective values
        results = []
        
        # 1. Initial Random Samples
        for i in range(num_initial):
            logger.info(f"Bayesian initial sample {i+1}/{num_initial}")
            
            params = self.optimizer.parameter_space.sample_random()
            params_array = self.optimizer.parameter_space.params_to_array(params)
            
            backtest_result = await self.optimizer.run_backtest(backtest_config, params)
            objective_value = self.optimizer.get_objective_value(backtest_result)
            
            X.append(params_array)
            y.append(objective_value)
            
            results.append(OptimizationResult(
                params=params,
                objective_value=objective_value,
                backtest_result=backtest_result
            ))
            
        # 2. Optimization Loop
        # Matern kernel handles non-smooth functions better than RBF
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        iterations_remaining = num_iterations - num_initial
        
        for i in range(iterations_remaining):
            logger.info(f"Bayesian optimization step {i+1}/{iterations_remaining}")
            
            # Fit GP to current data
            X_train = np.array(X)
            y_train = np.array(y)
            gp.fit(X_train, y_train)
            
            # Find next point to evaluate
            # We use a simple random sampling strategy to find max acquisition function
            # instead of gradient descent to avoid local optima and handle discrete params better
            candidates_params = [self.optimizer.parameter_space.sample_random() for _ in range(1000)]
            candidates_X = np.array([self.optimizer.parameter_space.params_to_array(p) for p in candidates_params])
            
            # Predict mean and std for candidates
            mu, sigma = gp.predict(candidates_X, return_std=True)
            
            # Expected Improvement (EI)
            best_y = np.max(y_train)
            with np.errstate(divide='warn'):
                imp = mu - best_y
                Z = imp / sigma
                # Normal CDF and PDF would be needed here, or specialized library
                # Simplified EI:
                ei = imp  # Exploitation
                # Add exploration bonus
                ei += 1.96 * sigma  # Upper Confidence Bound (UCB) simplified
            
            # Select best candidate
            best_idx = np.argmax(ei)
            next_params = candidates_params[best_idx]
            next_X = candidates_X[best_idx]
            
            # Evaluate
            backtest_result = await self.optimizer.run_backtest(backtest_config, next_params)
            objective_value = self.optimizer.get_objective_value(backtest_result)
            
            X.append(next_X)
            y.append(objective_value)
            results.append(OptimizationResult(
                params=next_params,
                objective_value=objective_value,
                backtest_result=backtest_result
            ))
            
            logger.info(f"Step outcome: {objective_value:.4f}")
            
        sorted_results = sorted(results, key=lambda r: r.objective_value, reverse=True)
        
        return OptimizationResults(
            method="bayesian",
            objective=self.optimizer.objective,
            results=sorted_results,
            best_params=sorted_results[0].params if sorted_results else {}
        )

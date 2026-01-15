"""
Tests for parameter optimization.
"""
import pytest
import asyncio
from typing import Any
from datetime import datetime
from dataclasses import dataclass, field

from src.backtesting.config import BacktestConfig
from src.backtesting.results import BacktestResults
from src.optimization.parameters import Parameter, ParameterSpace
from src.optimization.optimizer import Optimizer
from src.optimization.parallel import ParallelOptimizer


# ==================== Mocks ====================

@dataclass
class MockStrategy:
    """Mock strategy with parameters."""
    param_a: float = 0.5
    param_b: int = 10
    
    async def generate_signals(self, timestamp: datetime, prices: dict) -> list:
        return []

class MockBacktestRunner:
    """Mock runner that returns fake results based on params."""
    
    async def run_single(self, config, strategy) -> BacktestResults:
        # Create a fake objective function based on params
        # Simple parabolic function to test optimization
        
        # Access params from config or strategy
        # Optimizer sets them on both
        a = getattr(strategy, 'param_a', 0.5)
        b = getattr(strategy, 'param_b', 10)
        
        score = 10.0 - abs(a - 0.5) * 10 - abs(b - 10) * 0.5
        
        # DEBUG
        if abs(score - 10.0) < 0.1:
            print(f"DEBUG: Found optimal! a={a}, b={b}, score={score}")
        
        # Make sure BacktestResults has all required fields
        from datetime import timedelta
        return BacktestResults(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            duration=timedelta(0),
            initial_capital=1000.0,
            final_capital=1000.0,
            total_return=0.0,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=timedelta(0),
            sharpe_ratio=score,  # Objective
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            equity_curve=[],
            trades=[],
            signals_generated=0,
            signals_traded=0,
            performance_by_category={},
            performance_by_month={}
        )


# ==================== Tests ====================

class TestOptimization:
    
    @pytest.fixture
    def param_space(self):
        return ParameterSpace([
            Parameter("param_a", "float", 0.0, 1.0, 0.1),
            Parameter("param_b", "int", 5, 15, 1)
        ])
    
    @pytest.fixture
    def optimizer(self, param_space):
        runner = MockBacktestRunner()
        strategy = MockStrategy()
        return Optimizer(runner, strategy, param_space, objective="sharpe_ratio")
    
    @pytest.fixture
    def config(self):
        return BacktestConfig(datetime.now(), datetime.now())

    @pytest.mark.asyncio
    async def test_grid_search(self, optimizer, config):
        """Test grid search finds optimum."""
        results = await optimizer.optimize("grid", config)
        
        best = results.best_params
        assert abs(best["param_a"] - 0.5) < 0.001
        assert best["param_b"] == 10
        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_random_search(self, optimizer, config):
        """Test random search."""
        results = await optimizer.optimize("random", config, num_iterations=20)
        
        assert len(results.results) == 20
        assert "param_a" in results.best_params

    @pytest.mark.asyncio
    async def test_genetic_algorithm(self, optimizer, config):
        """Test genetic algorithm."""
        results = await optimizer.optimize("genetic", config, 
                                          population_size=10, 
                                          generations=3)
        
        assert len(results.results) > 0
        assert results.method == "genetic"

    @pytest.mark.asyncio
    async def test_parallel_optimization(self, optimizer, config):
        """Test parallel execution."""
        parallel = ParallelOptimizer(optimizer, num_workers=2)
        results = await parallel.parallel_grid_search(config)
        
        best = results.best_params
        assert abs(best["param_a"] - 0.5) < 0.001
        assert best["param_b"] == 10
        
    def test_parameter_space_sampling(self, param_space):
        """Test parameter sampling."""
        sample = param_space.sample_random()
        assert 0.0 <= sample["param_a"] <= 1.0
        assert 5 <= sample["param_b"] <= 15
        
        grid = param_space.get_grid_combinations()
        # 11 float steps (0.0 to 1.0) * 11 int steps (5 to 15) = 121
        assert len(grid) >= 100 

    def test_overfitting_detection(self, optimizer):
        from src.optimization.results import OptimizationResult, OptimizationResults
        
        in_sample = OptimizationResults("grid", "sharpe", 
                                       [OptimizationResult({}, 2.0, None)], {})
        out_sample = OptimizationResults("grid", "sharpe", 
                                        [OptimizationResult({}, 1.0, None)], {})
        
        analyzer = optimizer.optimizer_analyzer(in_sample) if hasattr(optimizer, "optimizer_analyzer") else None
        
        # Test direct logic if analyzer not attached to optimizer
        from src.optimization.results import OptimizationAnalyzer
        analyzer = OptimizationAnalyzer(in_sample)
        overfit = analyzer.detect_overfitting(in_sample, out_sample)
        
        assert overfit.is_overfit  # 50% degradation should trigger

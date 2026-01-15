"""
Backtest runner and orchestrator.
"""
import logging
import asyncio
import random
from copy import deepcopy
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any

from .config import BacktestConfig
from .backtester import Backtester, Strategy
from .data_provider import HistoricalDataProvider
from .results import BacktestResults

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Runs and manages backtest executions.
    
    Supports:
    - Single run
    - Walk-forward analysis
    - Monte Carlo simulation
    """
    
    def __init__(self, data_provider: HistoricalDataProvider):
        self.data_provider = data_provider
    
    async def run_single(
        self,
        config: BacktestConfig,
        strategy: Strategy
    ) -> BacktestResults:
        """Run a single backtest."""
        backtester = Backtester(config, self.data_provider, strategy)
        return await backtester.run()
    
    async def run_walk_forward(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        train_period: timedelta,
        test_period: timedelta,
        overlap: timedelta = timedelta(0)
    ) -> List[BacktestResults]:
        """
        Run walk-forward analysis.
        
        Splits time range into segments.
        Note: This effectively just runs multiple backtests on sequential periods.
        True WFA would involve optimization (training) on train_period and
        validating on test_period. Here we just return results for each window.
        """
        results = []
        current_start = config.start_date
        
        while current_start + train_period + test_period <= config.end_date:
            # Training window (optimization would happen here)
            train_start = current_start
            train_end = train_start + train_period
            
            # Test window
            test_start = train_end - overlap
            test_end = test_start + test_period
            
            logger.info(f"Walk-forward step: Test {test_start} to {test_end}")
            
            # Create config for this step
            step_config = deepcopy(config)
            step_config.start_date = test_start
            step_config.end_date = test_end
            
            # Run backtest for test period
            step_result = await self.run_single(step_config, strategy)
            results.append(step_result)
            
            # Move forward
            current_start += test_period
            
        return results
    
    async def run_monte_carlo(
        self,
        original_results: BacktestResults,
        num_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on trade results.
        
        Reshuffles trade sequence to analyze potential drawdown and return distributions.
        Does NOT re-run the backtest loops (which would be too slow),
        but simulates equity curves based on realized trade P&Ls.
        
        Returns:
            Dictionary with simulation stats (max DD at confidence interval, etc)
        """
        trades = sorted(
            [t for t in original_results.trades if t.realized_pnl is not None],
            key=lambda x: x.timestamp
        )
        pnl_sequence = [t.realized_pnl for t in trades]
        
        if not pnl_sequence:
            return {"error": "No closed trades to simulate"}
            
        final_equities = []
        max_drawdowns = []
        initial_capital = original_results.initial_capital
        
        for _ in range(num_simulations):
            # Shuffle P&L
            shuffled_pnl = list(pnl_sequence)
            random.shuffle(shuffled_pnl)
            
            # Reconstruct equity curve
            equity = [initial_capital]
            peak = initial_capital
            max_dd = 0.0
            
            for pnl in shuffled_pnl:
                current = equity[-1] + pnl
                equity.append(current)
                
                if current > peak:
                    peak = current
                
                dd = (peak - current) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                
            final_equities.append(equity[-1])
            max_drawdowns.append(max_dd)
            
        # Calculate stats
        final_equities.sort()
        max_drawdowns.sort()
        
        idx = int(num_simulations * (1 - confidence_level))
        
        return {
            "num_simulations": num_simulations,
            "mean_return": sum(final_equities) / num_simulations - initial_capital,
            "median_return": final_equities[num_simulations // 2] - initial_capital,
            "worst_case_drawdown": max_drawdowns[-1],
            f"drawdown_{int(confidence_level*100)}_conf": max_drawdowns[int(num_simulations * confidence_level)],
            f"min_equity_{int(confidence_level*100)}_conf": final_equities[idx],
        }

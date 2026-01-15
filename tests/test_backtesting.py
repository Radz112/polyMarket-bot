"""
Tests for backtesting framework.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.backtesting.config import (
    BacktestConfig, SlippageModel, FeeModel
)
from src.backtesting.data_provider import (
    HistoricalDataProvider, create_mock_dataset
)
from src.backtesting.backtester import Backtester
from src.backtesting.runner import BacktestRunner


# ==================== Mocks ====================

class MockStrategy:
    """Simple strategy for testing."""
    
    def __init__(self, should_trade: bool = True):
        self.should_trade = should_trade
        
    async def generate_signals(self, timestamp: datetime, prices: dict) -> list:
        if not self.should_trade:
            return []
            
        signals = []
        for market_id, price in prices.items():
            # Simple mean reversion strategy
            # Buy if price < 0.4, Sell/Short not implemented fully yet but we buy YES
            if price < 0.45:
                signals.append({
                    "id": f"sig_{market_id}_{timestamp.timestamp()}",
                    "market_id": market_id,
                    "market_name": f"Market {market_id}",
                    "side": "YES",
                    "score": 85,
                    "type": "value"
                })
        return signals


# ==================== Tests ====================

class TestBacktesting:
    
    @pytest.mark.asyncio
    async def test_full_backtest_run(self):
        """Test a complete backtest run."""
        # 1. Setup Data
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        mock_data = create_mock_dataset(
            num_markets=5,
            start_date=start_date,
            end_date=end_date,
            resolution_rate=0.5
        )
        
        provider = HistoricalDataProvider()
        provider.load_mock_data(mock_data)
        
        # 2. Setup Config
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            time_step=timedelta(hours=4),  # Fast steps
            slippage_model=SlippageModel.FIXED,
            fee_model=FeeModel.PERCENTAGE,
        )
        
        # 3. Setup Strategy
        strategy = MockStrategy(should_trade=True)
        
        # 4. Run Backtester
        backtester = Backtester(config, provider, strategy)
        results = await backtester.run()
        
        # 5. Verify Results
        assert results is not None
        assert results.total_trades >= 0
        assert len(results.equity_curve) > 0
        assert results.final_capital != 0
        
        # Should have generated signals
        assert results.signals_generated > 0
        
        print(f"\nBacktest Results:")
        print(f"Trades: {results.total_trades}")
        print(f"Return: {results.total_return:.2f} ({results.total_return_pct:.2f}%)")
        print(f"Final Capital: {results.final_capital:.2f}")

    @pytest.mark.asyncio
    async def test_runner_walk_forward(self):
        """Test walk-forward analysis."""
        # Setup Data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        mock_data = create_mock_dataset(
            num_markets=5,
            start_date=start_date,
            end_date=end_date
        )
        
        provider = HistoricalDataProvider()
        provider.load_mock_data(mock_data)
        
        runner = BacktestRunner(provider)
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            time_step=timedelta(hours=6)
        )
        strategy = MockStrategy()
        
        results_list = await runner.run_walk_forward(
            config,
            strategy,
            train_period=timedelta(days=7),
            test_period=timedelta(days=3)
        )
        
        assert len(results_list) > 0
        assert all(r.duration == timedelta(days=3) for r in results_list)

    @pytest.mark.asyncio
    async def test_monte_carlo(self):
        """Test Monte Carlo simulation."""
        # Create a fake result with some trades
        from src.backtesting.results import BacktestResults, SimulatedTrade
        
        trades = []
        for i in range(20):
            trades.append(SimulatedTrade(
                id=str(i),
                market_id="m1",
                market_name="Test",
                side="YES",
                action="SELL",
                size=100,
                price=0.5,
                fees=1,
                timestamp=datetime.now(),
                realized_pnl=10 if i % 2 == 0 else -5  # Win/Loss mix
            ))
            
        mock_results = BacktestResults(
            config=BacktestConfig(datetime.now(), datetime.now()),
            start_date=datetime.now(),
            end_date=datetime.now(),
            duration=timedelta(0),
            initial_capital=1000.0,
            final_capital=1100.0,
            total_return=100.0,
            total_return_pct=10.0,
            total_trades=20,
            winning_trades=10,
            losing_trades=10,
            win_rate=0.5,
            average_win=10,
            average_loss=-5,
            largest_win=10,
            largest_loss=-5,
            profit_factor=2.0,
            expectancy=2.5,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_duration=timedelta(0),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            equity_curve=[],
            trades=trades,
            signals_generated=0,
            signals_traded=0,
            performance_by_category={},
            performance_by_month={}
        )
        
        runner = BacktestRunner(HistoricalDataProvider())
        mc_results = await runner.run_monte_carlo(mock_results, num_simulations=100)
        
        assert "mean_return" in mc_results
        assert "worst_case_drawdown" in mc_results

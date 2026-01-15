"""
Backtester engine.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Protocol

from .config import BacktestConfig
from .data_provider import HistoricalDataProvider
from .portfolio import SimulatedPortfolio
from .results import BacktestResults, BacktestAnalyzer

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Protocol for strategies used in backtesting."""
    
    async def generate_signals(
        self,
        timestamp: datetime,
        prices: dict
    ) -> list:
        """Generate signals based on current state."""
        ...


class Backtester:
    """
    Main backtesting engine.
    
    Orchestrates the simulation loop:
    1. Iterates through time steps
    2. Updates data
    3. Generates signals
    4. Executes trades
    5. Tracks results
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        data_provider: HistoricalDataProvider,
        strategy: Strategy
    ):
        self.config = config
        self.data_provider = data_provider
        self.strategy = strategy
        self.portfolio = SimulatedPortfolio(config.initial_capital, config)
        self.results: Optional[BacktestResults] = None
        
        self.signals_generated = 0
        self.signals_traded = 0
    
    async def run(self) -> BacktestResults:
        """Run the backtest over the configured date range."""
        start_time = datetime.now()
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Ensure data is loaded
        market_ids = self.data_provider.get_market_ids()
        if not self.data_provider.is_loaded:
            await self.data_provider.load_data(
                market_ids,
                self.config.start_date,
                self.config.end_date
            )
        
        current_time = self.config.start_date
        step_size = self.config.time_step
        
        while current_time <= self.config.end_date:
            await self.run_step(current_time)
            
            # Check halt conditions
            if self.portfolio.max_drawdown > self.config.max_drawdown_halt:
                logger.warning(f"Max drawdown {self.portfolio.max_drawdown:.1%} exceeded halt limit")
                break
                
            current_time += step_size
        
        duration = datetime.now() - start_time
        logger.info(f"Backtest completed in {duration.total_seconds():.1f}s")
        
        # Generate results
        analyzer = BacktestAnalyzer(
            config=self.config,
            equity_curve=self.portfolio.snapshots,
            trades=self.portfolio.trades,
            signals_generated=self.signals_generated,
            signals_traded=self.signals_traded
        )
        self.results = analyzer.analyze()
        return self.results
    
    async def run_step(self, timestamp: datetime):
        """
        Execute single backtest step.
        """
        # 1. Update market data / Get prices
        all_prices = self.data_provider.get_all_prices_at(timestamp)
        prices_map = {mid: p.yes_price for mid, p in all_prices.items()}
        
        # 2. Check for market resolutions
        resolutions = self.data_provider.get_resolutions_before(timestamp)
        for res in resolutions:
            self.portfolio.resolve_market(
                res.market_id,
                res.outcome,
                timestamp
            )
        
        # 3. Update portfolio valuations
        self.portfolio.update_positions(prices_map)
        
        # 4. Generate signals
        # Pass context/simulated state to strategy?
        # For now assuming strategy handles its own state or just needs prices
        signals = await self.strategy.generate_signals(timestamp, prices_map)
        self.signals_generated += len(signals)
        
        # 5. Execute trades based on signals
        for signal in signals:
            # Signal expected format: 
            # {market_id, score, type, side, market_name...}
            
            # Simple filtering logic
            if signal.get('score', 0) < self.config.signal_threshold:
                continue
                
            market_id = signal['market_id']
            price = prices_map.get(market_id)
            if price is None:
                continue
            
            if not self.portfolio.can_open_position(prices_map):
                continue
                
            # Determine size
            size = self.portfolio.get_position_size(prices_map)
            
            if not self.portfolio.can_buy(size, price):
                continue
            
            trade = self.portfolio.execute_buy(
                market_id=market_id,
                market_name=signal.get('market_name', 'Unknown'),
                side=signal.get('side', 'YES'),
                size=size,
                price=price,
                timestamp=timestamp,
                signal_id=signal.get('id'),
                signal_score=signal.get('score')
            )
            
            if trade:
                self.signals_traded += 1
        
        # 6. Take snapshot
        self.portfolio.take_snapshot(timestamp, prices_map)

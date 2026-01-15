"""
Paper trading execution engine.

Executes simulated trades based on signals without using real money.
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, List

from src.config.settings import Config
from src.database.postgres import DatabaseManager
from src.database.redis_cache import CacheManager
from src.models.orderbook import Orderbook
from src.signals.scoring.types import ScoredSignal
from src.execution.paper.types import (
    PaperPosition,
    PaperTrade,
    PnLSummary,
    PortfolioState,
)
from src.execution.paper.fill_simulator import FillSimulator

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading execution engine.
    
    Simulates trade execution with:
    - Realistic fill simulation (slippage, fees)
    - Position tracking with P&L
    - Balance management
    - Trade history logging
    """
    
    def __init__(
        self,
        config: Config,
        db: Optional[DatabaseManager] = None,
        cache: Optional[CacheManager] = None,
        fill_simulator: Optional[FillSimulator] = None
    ):
        """
        Initialize the paper trader.
        
        Args:
            config: Application configuration
            db: Database manager for persistence
            cache: Cache manager for orderbook access
            fill_simulator: Optional custom fill simulator
        """
        self.config = config
        self.db = db
        self.cache = cache
        
        # Initialize balance
        self.initial_balance = config.paper_trading_balance
        self.balance = self.initial_balance
        
        # Track positions and trades
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        
        # Fill simulator
        self.fill_simulator = fill_simulator or FillSimulator(
            slippage_model="realistic",
            fee_pct=config.trading_fees_pct
        )
        
        logger.info(
            f"Initialized PaperTrader with balance=${self.initial_balance:.2f}, "
            f"fees={config.trading_fees_pct*100:.1f}%"
        )
    
    async def execute_signal(
        self,
        signal: ScoredSignal,
        size: Optional[float] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a paper trade based on a scored signal.
        
        Args:
            signal: The scored signal to execute
            size: Optional position size override
            
        Returns:
            PaperTrade if executed, None if not possible
        """
        # Get recommended trade parameters from signal
        if size is None:
            size = signal.recommended_size
        
        if size <= 0:
            logger.warning(f"Signal {signal.divergence.id} has no recommended size")
            return None
        
        price = signal.recommended_price
        if price <= 0:
            logger.warning(f"Signal {signal.divergence.id} has no recommended price")
            return None
        
        # Determine trade direction from signal
        # The divergence metadata should indicate which market/side to trade
        market_id = signal.divergence.market_ids[0] if signal.divergence.market_ids else None
        if not market_id:
            logger.warning(f"Signal {signal.divergence.id} has no market ID")
            return None
        
        # Get market name from metadata or use market_id
        market_name = signal.divergence.metadata.get("market_name", market_id)
        
        # Determine side from signal metadata (default to YES)
        side = signal.divergence.metadata.get("trade_side", "YES")
        
        # Execute the buy
        return await self.buy(
            market_id=market_id,
            side=side,
            size=size,
            limit_price=price,
            market_name=market_name,
            signal_id=signal.divergence.id,
            signal_score=signal.overall_score
        )
    
    async def buy(
        self,
        market_id: str,
        side: str,  # "YES" or "NO"
        size: float,
        limit_price: Optional[float] = None,
        market_name: Optional[str] = None,
        signal_id: Optional[str] = None,
        signal_score: Optional[float] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a paper buy order.
        
        Args:
            market_id: Market to trade
            side: "YES" or "NO"
            size: Number of shares to buy
            limit_price: Optional limit price
            market_name: Optional market name for display
            signal_id: Optional signal that triggered trade
            signal_score: Optional signal score
            
        Returns:
            PaperTrade if executed, None if not possible
        """
        # Validate
        if size <= 0:
            logger.warning("Cannot buy zero or negative size")
            return None
        
        # Get orderbook for realistic fill
        orderbook = await self._get_orderbook(market_id)
        
        # If no orderbook, use limit price as fill price
        if orderbook is None:
            if limit_price is None:
                logger.warning(f"No orderbook and no limit price for {market_id}")
                return None
            fill_price = limit_price
            fill_size = size
            slippage = 0.0
            fees = size * limit_price * self.fill_simulator.fee_pct
        else:
            # Simulate fill
            result = self.fill_simulator.simulate_fill(
                orderbook=orderbook,
                side="buy",
                size=size,
                limit_price=limit_price
            )
            
            if result.filled_size == 0:
                logger.warning(f"Could not fill buy order for {market_id}")
                return None
            
            fill_price = result.average_price
            fill_size = result.filled_size
            slippage = result.slippage
            fees = result.fees
        
        # Check sufficient balance
        total_cost = fill_size * fill_price + fees
        if total_cost > self.balance:
            logger.warning(
                f"Insufficient balance for {market_id}: "
                f"need ${total_cost:.2f}, have ${self.balance:.2f}"
            )
            return None
        
        # Deduct from balance
        self.balance -= total_cost
        self.total_fees += fees
        
        # Create or update position
        position_id = f"pos_{market_id}_{side}"
        if position_id in self.positions:
            # Average into existing position
            pos = self.positions[position_id]
            old_cost = pos.size * pos.entry_price
            new_cost = fill_size * fill_price
            new_size = pos.size + fill_size
            pos.size = new_size
            pos.entry_price = (old_cost + new_cost) / new_size
            pos.current_price = fill_price
        else:
            # Create new position
            self.positions[position_id] = PaperPosition(
                id=position_id,
                market_id=market_id,
                market_name=market_name or market_id,
                side=side,
                size=fill_size,
                entry_price=fill_price,
                current_price=fill_price,
                signal_id=signal_id,
                signal_score=signal_score
            )
        
        # Create trade record
        trade = PaperTrade(
            id=f"trade_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            market_name=market_name or market_id,
            side=side,
            action="BUY",
            size=fill_size,
            price=fill_price,
            fees=fees,
            total=total_cost,
            signal_id=signal_id,
            signal_score=signal_score,
            position_id=position_id
        )
        self.trades.append(trade)
        
        logger.info(
            f"BUY {fill_size:.4f} {side} @ ${fill_price:.4f} "
            f"(slippage: ${slippage:.4f}, fees: ${fees:.4f}) - {market_name}"
        )
        
        return trade
    
    async def sell(
        self,
        market_id: str,
        side: str,  # "YES" or "NO"
        size: float,
        limit_price: Optional[float] = None,
        market_name: Optional[str] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a paper sell order.
        
        Args:
            market_id: Market to trade
            side: "YES" or "NO"
            size: Number of shares to sell
            limit_price: Optional limit price
            market_name: Optional market name for display
            
        Returns:
            PaperTrade if executed, None if not possible
        """
        # Validate
        if size <= 0:
            logger.warning("Cannot sell zero or negative size")
            return None
        
        # Check position exists
        position_id = f"pos_{market_id}_{side}"
        if position_id not in self.positions:
            logger.warning(f"No position to sell for {market_id} {side}")
            return None
        
        pos = self.positions[position_id]
        if size > pos.size:
            logger.warning(
                f"Cannot sell {size} shares, only have {pos.size}"
            )
            size = pos.size  # Sell what we have
        
        # Get orderbook for realistic fill
        orderbook = await self._get_orderbook(market_id)
        
        # If no orderbook, use limit price or current price
        if orderbook is None:
            if limit_price is not None:
                fill_price = limit_price
            else:
                fill_price = pos.current_price
            fill_size = size
            fees = size * fill_price * self.fill_simulator.fee_pct
        else:
            # Simulate fill
            result = self.fill_simulator.simulate_fill(
                orderbook=orderbook,
                side="sell",
                size=size,
                limit_price=limit_price
            )
            
            if result.filled_size == 0:
                logger.warning(f"Could not fill sell order for {market_id}")
                return None
            
            fill_price = result.average_price
            fill_size = result.filled_size
            fees = result.fees
        
        # Calculate P&L for this sale
        cost_basis = fill_size * pos.entry_price
        proceeds = fill_size * fill_price - fees
        trade_pnl = proceeds - cost_basis
        
        # Update balance
        self.balance += proceeds
        self.total_fees += fees
        self.realized_pnl += trade_pnl
        
        # Update position
        pos.size -= fill_size
        pos.current_price = fill_price
        
        # Remove position if fully closed
        if pos.size <= 0.0001:  # Small epsilon for float comparison
            del self.positions[position_id]
            holding_period = datetime.utcnow() - pos.opened_at
        else:
            holding_period = None
        
        # Create trade record
        trade = PaperTrade(
            id=f"trade_{uuid.uuid4().hex[:8]}",
            market_id=market_id,
            market_name=market_name or pos.market_name,
            side=side,
            action="SELL",
            size=fill_size,
            price=fill_price,
            fees=fees,
            total=proceeds,
            position_id=position_id,
            realized_pnl=trade_pnl,
            holding_period=holding_period
        )
        self.trades.append(trade)
        
        logger.info(
            f"SELL {fill_size:.4f} {side} @ ${fill_price:.4f} "
            f"(P&L: ${trade_pnl:.4f}, fees: ${fees:.4f}) - {market_name or pos.market_name}"
        )
        
        return trade
    
    async def close_position(
        self,
        position_id: str
    ) -> Optional[PaperTrade]:
        """
        Close an entire position at market price.
        
        Args:
            position_id: ID of position to close
            
        Returns:
            PaperTrade if closed, None if not possible
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return None
        
        pos = self.positions[position_id]
        return await self.sell(
            market_id=pos.market_id,
            side=pos.side,
            size=pos.size,
            market_name=pos.market_name
        )
    
    async def close_all_positions(self) -> List[PaperTrade]:
        """Close all open positions."""
        trades = []
        position_ids = list(self.positions.keys())
        
        for position_id in position_ids:
            trade = await self.close_position(position_id)
            if trade:
                trades.append(trade)
        
        return trades
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Cash balance plus mark-to-market value of all positions
        """
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.balance + positions_value
    
    def get_pnl(self) -> PnLSummary:
        """
        Calculate current P&L summary.
        
        Returns:
            PnLSummary with realized, unrealized, and total P&L
        """
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        total_pnl = self.realized_pnl + unrealized_pnl
        return_pct = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return PnLSummary(
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            total_fees=self.total_fees,
            return_pct=return_pct
        )
    
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state snapshot.
        
        Returns:
            Complete portfolio state
        """
        pnl = self.get_pnl()
        
        return PortfolioState(
            timestamp=datetime.utcnow(),
            cash_balance=self.balance,
            positions=list(self.positions.values()),
            total_value=self.get_portfolio_value(),
            unrealized_pnl=pnl.unrealized_pnl,
            realized_pnl=pnl.realized_pnl,
            total_pnl=pnl.total_pnl,
            return_pct=pnl.return_pct
        )
    
    def update_position_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for all positions.
        
        Args:
            prices: Dict mapping market_id to current price
        """
        for pos in self.positions.values():
            if pos.market_id in prices:
                pos.current_price = prices[pos.market_id]
    
    async def _get_orderbook(self, market_id: str) -> Optional[Orderbook]:
        """Get orderbook from cache."""
        if self.cache is None:
            return None
        
        try:
            return await self.cache.get_orderbook(market_id)
        except Exception as e:
            logger.warning(f"Failed to get orderbook for {market_id}: {e}")
            return None
    
    def reset(self) -> None:
        """Reset paper trader to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        logger.info(f"Paper trader reset to ${self.initial_balance:.2f}")

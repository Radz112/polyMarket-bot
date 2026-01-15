"""
Simulated portfolio for backtesting.
"""
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

from .config import (
    BacktestConfig, SimulatedPosition, SimulatedTrade,
    PortfolioSnapshot, SlippageModel, FeeModel
)

logger = logging.getLogger(__name__)


class SimulatedPortfolio:
    """
    Simulates a trading portfolio for backtesting.
    
    Handles:
    - Position tracking
    - Trade execution with slippage/fees
    - P&L calculation
    - Portfolio snapshots
    """
    
    def __init__(self, initial_capital: float, config: BacktestConfig):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.config = config
        
        self.positions: Dict[str, SimulatedPosition] = {}
        self.trades: List[SimulatedTrade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.total_fees = 0.0
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            prices: Dict mapping market_id to current YES price
        """
        positions_value = 0.0
        for key, pos in self.positions.items():
            market_id = pos.market_id
            if pos.side == "YES":
                price = prices.get(market_id, pos.current_price)
            else:
                price = 1 - prices.get(market_id, 1 - pos.current_price)
            positions_value += pos.size * price
        
        return self.cash + positions_value
    
    def get_positions_value(self, prices: Dict[str, float]) -> float:
        """Calculate total positions value."""
        value = 0.0
        for pos in self.positions.values():
            if pos.side == "YES":
                price = prices.get(pos.market_id, pos.current_price)
            else:
                price = 1 - prices.get(pos.market_id, 1 - pos.current_price)
            value += pos.size * price
        return value
    
    def can_buy(self, size: float, price: float) -> bool:
        """Check if we have enough cash to buy."""
        total_cost = size + self._calculate_fees(size, price)
        return self.cash >= total_cost
    
    def can_open_position(self, prices: Dict[str, float]) -> bool:
        """Check if we can open a new position (max positions check)."""
        return len(self.positions) < self.config.max_positions
    
    def get_position_size(self, prices: Dict[str, float]) -> float:
        """Calculate position size based on config."""
        method = self.config.position_size_method
        value = self.config.position_size_value
        total_value = self.get_total_value(prices)
        
        if method.value == "fixed":
            return min(value, self.cash * 0.95)  # Leave some buffer
        elif method.value == "percent":
            max_size = total_value * (value / 100)
            return min(max_size, self.cash * 0.95)
        elif method.value == "kelly":
            # Simplified Kelly - would need win rate and odds
            return min(value, self.cash * 0.95)
        
        return value
    
    def execute_buy(
        self,
        market_id: str,
        market_name: str,
        side: str,
        size: float,
        price: float,
        timestamp: datetime,
        signal_id: Optional[str] = None,
        signal_score: Optional[float] = None
    ) -> Optional[SimulatedTrade]:
        """
        Execute a simulated buy order.
        
        Args:
            market_id: Market ID
            market_name: Human readable market name
            side: "YES" or "NO"
            size: Dollar size to buy
            price: Current price
            timestamp: Execution time
            signal_id: Optional signal that triggered this trade
            signal_score: Optional signal score
            
        Returns:
            SimulatedTrade if successful, None if insufficient funds
        """
        # Apply slippage
        fill_price = self._apply_slippage(price, "buy")
        
        # Calculate shares and fees
        shares = size / fill_price
        fees = self._calculate_fees(size, fill_price)
        
        total_cost = size + fees
        
        # Check funds
        if total_cost > self.cash:
            logger.debug(f"Insufficient funds for buy: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return None
        
        # Deduct cash
        self.cash -= total_cost
        self.total_fees += fees
        
        # Update or create position
        position_key = f"{market_id}_{side}"
        if position_key in self.positions:
            pos = self.positions[position_key]
            old_value = pos.size * pos.entry_price
            new_value = shares * fill_price
            new_shares = pos.size + shares
            pos.entry_price = (old_value + new_value) / new_shares
            pos.size = new_shares
        else:
            self.positions[position_key] = SimulatedPosition(
                market_id=market_id,
                side=side,
                size=shares,
                entry_price=fill_price,
                entry_time=timestamp,
                current_price=fill_price,
            )
        
        # Record trade
        trade = SimulatedTrade(
            id=str(uuid.uuid4())[:8],
            market_id=market_id,
            market_name=market_name,
            side=side,
            action="BUY",
            size=size,
            price=fill_price,
            fees=fees,
            timestamp=timestamp,
            signal_id=signal_id,
            signal_score=signal_score,
        )
        self.trades.append(trade)
        
        logger.debug(f"BUY {side} @ {fill_price:.4f}, size=${size:.2f}, fees=${fees:.2f}")
        return trade
    
    def execute_sell(
        self,
        market_id: str,
        side: str,
        size: Optional[float],
        price: float,
        timestamp: datetime
    ) -> Optional[SimulatedTrade]:
        """
        Execute a simulated sell order.
        
        Args:
            market_id: Market ID
            side: "YES" or "NO"
            size: Dollar size to sell (None for full position)
            price: Current price
            timestamp: Execution time
            
        Returns:
            SimulatedTrade if successful
        """
        position_key = f"{market_id}_{side}"
        if position_key not in self.positions:
            logger.debug(f"No position to sell: {position_key}")
            return None
        
        pos = self.positions[position_key]
        
        # Apply slippage
        fill_price = self._apply_slippage(price, "sell")
        
        # Determine sell size
        if size is None:
            shares_to_sell = pos.size
        else:
            shares_to_sell = min(size / fill_price, pos.size)
        
        sell_value = shares_to_sell * fill_price
        fees = self._calculate_fees(sell_value, fill_price)
        
        # Calculate P&L
        entry_value = shares_to_sell * pos.entry_price
        realized_pnl = sell_value - entry_value - fees
        
        # Add to cash
        self.cash += sell_value - fees
        self.total_fees += fees
        
        # Update position
        if shares_to_sell >= pos.size:
            del self.positions[position_key]
        else:
            pos.size -= shares_to_sell
        
        # Get market name from previous trade
        market_name = next(
            (t.market_name for t in reversed(self.trades) if t.market_id == market_id),
            "Unknown"
        )
        
        # Record trade
        trade = SimulatedTrade(
            id=str(uuid.uuid4())[:8],
            market_id=market_id,
            market_name=market_name,
            side=side,
            action="SELL",
            size=sell_value,
            price=fill_price,
            fees=fees,
            timestamp=timestamp,
            realized_pnl=realized_pnl,
        )
        self.trades.append(trade)
        
        logger.debug(f"SELL {side} @ {fill_price:.4f}, size=${sell_value:.2f}, P&L=${realized_pnl:.2f}")
        return trade
    
    def resolve_market(
        self,
        market_id: str,
        outcome: str,
        timestamp: datetime
    ) -> List[SimulatedTrade]:
        """
        Handle market resolution.
        
        Args:
            market_id: Market that resolved
            outcome: "YES" or "NO"
            timestamp: Resolution time
            
        Returns:
            List of resolution trades
        """
        resolution_trades = []
        
        for key in list(self.positions.keys()):
            pos = self.positions[key]
            if pos.market_id != market_id:
                continue
            
            # Settlement price
            if pos.side == outcome:
                settlement_price = 1.0  # Won
            else:
                settlement_price = 0.0  # Lost
            
            settlement_value = pos.size * settlement_price
            realized_pnl = settlement_value - (pos.size * pos.entry_price)
            
            # Add to cash
            self.cash += settlement_value
            
            # Get market name
            market_name = next(
                (t.market_name for t in reversed(self.trades) if t.market_id == market_id),
                "Unknown"
            )
            
            # Record trade
            trade = SimulatedTrade(
                id=str(uuid.uuid4())[:8],
                market_id=market_id,
                market_name=market_name,
                side=pos.side,
                action="RESOLVE",
                size=pos.size,
                price=settlement_price,
                fees=0,
                timestamp=timestamp,
                realized_pnl=realized_pnl,
            )
            self.trades.append(trade)
            resolution_trades.append(trade)
            
            # Remove position
            del self.positions[key]
            
            won = "WON" if pos.side == outcome else "LOST"
            logger.debug(f"RESOLVE {pos.side} {won}, P&L=${realized_pnl:.2f}")
        
        return resolution_trades
    
    def update_positions(self, prices: Dict[str, float]):
        """Update position current prices."""
        for pos in self.positions.values():
            if pos.side == "YES":
                pos.current_price = prices.get(pos.market_id, pos.current_price)
            else:
                pos.current_price = 1 - prices.get(pos.market_id, 1 - pos.current_price)
    
    def take_snapshot(self, timestamp: datetime, prices: Dict[str, float]):
        """Record portfolio state."""
        total_value = self.get_total_value(prices)
        positions_value = self.get_positions_value(prices)
        
        # Track drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            num_positions=len(self.positions),
            drawdown=drawdown,
        )
        self.snapshots.append(snapshot)
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage model to price."""
        model = self.config.slippage_model
        bps = self.config.slippage_bps
        
        if model == SlippageModel.NONE:
            return price
        elif model == SlippageModel.FIXED:
            slippage = bps / 10000
            if direction == "buy":
                return min(0.99, price * (1 + slippage))
            else:
                return max(0.01, price * (1 - slippage))
        elif model == SlippageModel.PERCENTAGE:
            slippage = bps / 10000
            if direction == "buy":
                return min(0.99, price * (1 + slippage))
            else:
                return max(0.01, price * (1 - slippage))
        
        return price
    
    def _calculate_fees(self, size: float, price: float) -> float:
        """Calculate fees for trade."""
        model = self.config.fee_model
        bps = self.config.fee_bps
        
        if model == FeeModel.NONE:
            return 0.0
        elif model == FeeModel.FIXED:
            return bps / 100  # Treat as fixed dollar amount
        elif model == FeeModel.PERCENTAGE:
            return size * (bps / 10000)
        
        return 0.0
    
    def get_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(t.realized_pnl or 0 for t in self.trades)
    
    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Get total unrealized P&L."""
        self.update_positions(prices)
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def get_trade_summary(self) -> dict:
        """Get summary of trades."""
        closed_trades = [t for t in self.trades if t.realized_pnl is not None]
        
        if not closed_trades:
            return {
                "total_trades": len(self.trades),
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }
        
        wins = sum(1 for t in closed_trades if t.realized_pnl > 0)
        losses = sum(1 for t in closed_trades if t.realized_pnl <= 0)
        
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(closed_trades) if closed_trades else 0.0,
        }

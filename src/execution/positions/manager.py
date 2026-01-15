"""
Position manager for tracking and managing trading positions.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from src.database.postgres import DatabaseManager
from src.database.redis_cache import CacheManager
from src.execution.positions.types import (
    Position,
    PositionStatus,
    ClosedPosition,
)

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions with real-time updates.
    
    Provides:
    - Position lifecycle (open/reduce/close)
    - Price updates from market data
    - Exposure calculations
    - Fast lookups by market/side
    """
    
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        cache: Optional[CacheManager] = None
    ):
        """
        Initialize position manager.
        
        Args:
            db: Database manager for persistence
            cache: Cache manager for real-time prices
        """
        self.db = db
        self.cache = cache
        
        # In-memory position storage for fast access
        self.positions: Dict[str, Position] = {}
        
        # Indexes for fast lookups
        self._by_market: Dict[str, List[str]] = {}  # market_id -> [position_ids]
        self._by_category: Dict[str, List[str]] = {}  # category -> [position_ids]
        
        # Closed positions history
        self.closed_positions: List[ClosedPosition] = []
    
    async def load_positions(self) -> int:
        """
        Load open positions from database.
        
        Returns:
            Number of positions loaded
        """
        if self.db is None:
            logger.warning("No database for loading positions")
            return 0
        
        try:
            from sqlalchemy import text
            
            async with self.db._session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT id, market_id, market_name, side, size, entry_price, 
                               opened_at, signal_id, signal_score, status
                        FROM paper_positions
                        WHERE status = 'open'
                    """)
                )
                rows = result.fetchall()
                
                for row in rows:
                    position = Position(
                        id=row[0],
                        market_id=row[1],
                        market_name=row[2] or row[1],
                        category="Unknown",  # Would need market lookup
                        side=row[3],
                        size=row[4],
                        entry_price=row[5],
                        entry_time=row[6],
                        current_price=row[5],  # Start with entry price
                        signal_id=row[7],
                        signal_score=row[8],
                        status=PositionStatus(row[9]) if row[9] else PositionStatus.OPEN
                    )
                    self._add_position(position)
                
                logger.info(f"Loaded {len(rows)} positions from database")
                return len(rows)
                
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return 0
    
    async def open_position(
        self,
        market_id: str,
        side: str,
        size: float,
        entry_price: float,
        market_name: str = None,
        category: str = "Unknown",
        signal_id: str = None,
        signal_score: float = None
    ) -> Position:
        """
        Open new position or add to existing.
        
        If position exists in same market/side:
        - Average the entry price
        - Add to size
        
        Args:
            market_id: Market identifier
            side: "YES" or "NO"
            size: Position size
            entry_price: Entry price
            market_name: Human-readable market name
            category: Market category
            signal_id: Optional triggering signal
            signal_score: Optional signal score
            
        Returns:
            New or updated Position
        """
        # Check for existing position
        existing = self.get_position(market_id, side)
        
        if existing:
            # Average into existing position
            old_cost = existing.size * existing.entry_price
            new_cost = size * entry_price
            new_size = existing.size + size
            new_entry = (old_cost + new_cost) / new_size
            
            existing.size = new_size
            existing.entry_price = new_entry
            existing.current_price = entry_price
            existing.last_update = datetime.utcnow()
            
            logger.info(
                f"Averaged into position {existing.id}: "
                f"size={new_size:.4f}, entry={new_entry:.4f}"
            )
            return existing
        
        # Create new position
        position_id = f"pos_{market_id}_{side}_{uuid.uuid4().hex[:6]}"
        position = Position(
            id=position_id,
            market_id=market_id,
            market_name=market_name or market_id,
            category=category,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            current_price=entry_price,
            signal_id=signal_id,
            signal_score=signal_score
        )
        
        self._add_position(position)
        logger.info(
            f"Opened position {position_id}: {side} {size:.4f} @ {entry_price:.4f}"
        )
        
        return position
    
    async def reduce_position(
        self,
        position_id: str,
        size: float,
        exit_price: float
    ) -> tuple:
        """
        Reduce position size.
        
        Args:
            position_id: Position to reduce
            size: Amount to reduce by
            exit_price: Exit price for P&L calculation
            
        Returns:
            Tuple of (updated Position, realized P&L)
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        if size > position.size:
            size = position.size  # Can't reduce more than we have
        
        # Calculate realized P&L for reduced portion
        cost_per_share = position.entry_price
        proceeds_per_share = exit_price
        realized_pnl = (proceeds_per_share - cost_per_share) * size
        
        # Update position
        position.size -= size
        position.current_price = exit_price
        position.last_update = datetime.utcnow()
        
        logger.info(
            f"Reduced position {position_id} by {size:.4f} @ {exit_price:.4f}, "
            f"P&L: ${realized_pnl:.4f}"
        )
        
        # If fully closed, remove position
        if position.size < 0.0001:
            await self.close_position(position_id, exit_price, "partial_close")
        
        return position, realized_pnl
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "manual"
    ) -> float:
        """
        Close entire position.
        
        Args:
            position_id: Position to close
            exit_price: Exit price
            exit_reason: Why closing ("signal", "stop_loss", "take_profit", "manual", "resolution")
            
        Returns:
            Realized P&L
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        # Calculate final P&L
        realized_pnl = (exit_price - position.entry_price) * position.size
        holding_period = datetime.utcnow() - position.entry_time
        
        # Create closed position record
        closed = ClosedPosition(
            position=position,
            exit_price=exit_price,
            exit_time=datetime.utcnow(),
            realized_pnl=realized_pnl,
            holding_period=holding_period,
            exit_reason=exit_reason
        )
        self.closed_positions.append(closed)
        
        # Update status and remove
        position.status = PositionStatus.CLOSED
        self._remove_position(position_id)
        
        logger.info(
            f"Closed position {position_id}: P&L=${realized_pnl:.4f}, "
            f"held for {holding_period}"
        )
        
        return realized_pnl
    
    async def update_prices(self) -> int:
        """
        Update all positions with current market prices.
        
        Returns:
            Number of positions updated
        """
        if self.cache is None:
            return 0
        
        updated = 0
        for position in self.positions.values():
            try:
                # Get orderbook for bid/ask
                orderbook = await self.cache.get_orderbook(position.market_id)
                if orderbook:
                    bid = orderbook.best_bid or position.current_price
                    ask = orderbook.best_ask or position.current_price
                    mid = orderbook.mid_price or position.current_price
                    position.update_price(mid, bid, ask)
                    updated += 1
                else:
                    # Try price snapshot
                    price = await self.cache.get_price(position.market_id)
                    if price:
                        # Use yes_price or no_price based on side
                        if position.side == "YES":
                            position.update_price(price.yes_price)
                        else:
                            position.update_price(price.no_price)
                        updated += 1
            except Exception as e:
                logger.warning(f"Failed to update price for {position.market_id}: {e}")
        
        return updated
    
    def update_position_price(
        self,
        market_id: str,
        price: float,
        bid: float = None,
        ask: float = None
    ) -> int:
        """
        Update price for all positions in a market.
        
        Args:
            market_id: Market to update
            price: Current price
            bid: Current bid
            ask: Current ask
            
        Returns:
            Number of positions updated
        """
        updated = 0
        for pos_id in self._by_market.get(market_id, []):
            if pos_id in self.positions:
                self.positions[pos_id].update_price(price, bid, ask)
                updated += 1
        return updated
    
    def get_position(self, market_id: str, side: str) -> Optional[Position]:
        """Get position by market and side."""
        for pos_id in self._by_market.get(market_id, []):
            pos = self.positions.get(pos_id)
            if pos and pos.side == side:
                return pos
        return None
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_positions_by_market(self, market_id: str) -> List[Position]:
        """Get all positions in a market."""
        return [
            self.positions[pid]
            for pid in self._by_market.get(market_id, [])
            if pid in self.positions
        ]
    
    def get_positions_by_category(self, category: str) -> List[Position]:
        """Get all positions in a category."""
        return [
            self.positions[pid]
            for pid in self._by_category.get(category, [])
            if pid in self.positions
        ]
    
    def get_total_exposure(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())
    
    def get_exposure_by_category(self) -> Dict[str, float]:
        """Exposure grouped by market category."""
        exposure = {}
        for position in self.positions.values():
            cat = position.category
            exposure[cat] = exposure.get(cat, 0) + position.market_value
        return exposure
    
    def get_total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def _add_position(self, position: Position) -> None:
        """Add position to storage and indexes."""
        self.positions[position.id] = position
        
        # Update market index
        if position.market_id not in self._by_market:
            self._by_market[position.market_id] = []
        self._by_market[position.market_id].append(position.id)
        
        # Update category index
        if position.category not in self._by_category:
            self._by_category[position.category] = []
        self._by_category[position.category].append(position.id)
    
    def _remove_position(self, position_id: str) -> None:
        """Remove position from storage and indexes."""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Remove from indexes
        if position.market_id in self._by_market:
            self._by_market[position.market_id] = [
                pid for pid in self._by_market[position.market_id]
                if pid != position_id
            ]
        
        if position.category in self._by_category:
            self._by_category[position.category] = [
                pid for pid in self._by_category[position.category]
                if pid != position_id
            ]
        
        del self.positions[position_id]
    
    def clear(self) -> None:
        """Clear all positions."""
        self.positions.clear()
        self._by_market.clear()
        self._by_category.clear()
        self.closed_positions.clear()

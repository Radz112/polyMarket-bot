"""
Persistence layer for paper trading.

Saves and loads paper trading state to/from database.
"""
import logging
from datetime import datetime
from typing import Optional, List

from src.database.postgres import DatabaseManager
from src.execution.paper.types import (
    PaperPosition,
    PaperTrade,
    PortfolioState,
    PortfolioSnapshot,
    PositionStatus,
)

logger = logging.getLogger(__name__)


class PaperTradingPersistence:
    """
    Handles persistence of paper trading state.
    
    Supports:
    - Saving/loading complete trader state
    - Recording individual trades
    - Saving portfolio snapshots for equity curve
    - Resetting to initial state
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        Initialize persistence layer.
        
        Args:
            db: Database manager for persistence
        """
        self.db = db
    
    async def save_state(
        self,
        balance: float,
        initial_balance: float,
        positions: dict,
        realized_pnl: float,
        total_fees: float
    ) -> bool:
        """
        Save current paper trading state to database.
        
        Args:
            balance: Current cash balance
            initial_balance: Initial balance
            positions: Dict of positions
            realized_pnl: Total realized P&L
            total_fees: Total fees paid
            
        Returns:
            True if saved successfully
        """
        if self.db is None:
            logger.warning("No database connection for persistence")
            return False
        
        try:
            async with self.db._session_factory() as session:
                # Update or insert portfolio record
                from sqlalchemy import text
                
                # Check if portfolio exists
                result = await session.execute(
                    text("SELECT id FROM paper_portfolio LIMIT 1")
                )
                existing = result.scalar()
                
                if existing:
                    await session.execute(
                        text("""
                            UPDATE paper_portfolio 
                            SET balance = :balance, 
                                initial_balance = :initial_balance,
                                realized_pnl = :realized_pnl,
                                total_fees = :total_fees,
                                updated_at = NOW()
                            WHERE id = :id
                        """),
                        {
                            "id": existing,
                            "balance": balance,
                            "initial_balance": initial_balance,
                            "realized_pnl": realized_pnl,
                            "total_fees": total_fees
                        }
                    )
                else:
                    await session.execute(
                        text("""
                            INSERT INTO paper_portfolio 
                            (balance, initial_balance, realized_pnl, total_fees)
                            VALUES (:balance, :initial_balance, :realized_pnl, :total_fees)
                        """),
                        {
                            "balance": balance,
                            "initial_balance": initial_balance,
                            "realized_pnl": realized_pnl,
                            "total_fees": total_fees
                        }
                    )
                
                await session.commit()
                logger.debug("Saved paper trading state")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save paper trading state: {e}")
            return False
    
    async def load_state(self) -> Optional[dict]:
        """
        Load paper trading state from database.
        
        Returns:
            Dict with balance, initial_balance, realized_pnl, total_fees
            or None if not found
        """
        if self.db is None:
            return None
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                result = await session.execute(
                    text("""
                        SELECT balance, initial_balance, realized_pnl, total_fees
                        FROM paper_portfolio
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """)
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "balance": row[0],
                        "initial_balance": row[1],
                        "realized_pnl": row[2] or 0.0,
                        "total_fees": row[3] or 0.0
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to load paper trading state: {e}")
            return None
    
    async def save_position(self, position: PaperPosition) -> bool:
        """
        Save a position to database.
        
        Args:
            position: Position to save
            
        Returns:
            True if saved successfully
        """
        if self.db is None:
            return False
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                await session.execute(
                    text("""
                        INSERT INTO paper_positions 
                        (id, market_id, market_name, side, size, entry_price, opened_at, status, signal_id, signal_score)
                        VALUES (:id, :market_id, :market_name, :side, :size, :entry_price, :opened_at, :status, :signal_id, :signal_score)
                        ON CONFLICT (id) DO UPDATE SET
                            size = :size,
                            entry_price = :entry_price,
                            status = :status
                    """),
                    {
                        "id": position.id,
                        "market_id": position.market_id,
                        "market_name": position.market_name,
                        "side": position.side,
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "opened_at": position.opened_at,
                        "status": PositionStatus.OPEN.value,
                        "signal_id": position.signal_id,
                        "signal_score": position.signal_score
                    }
                )
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            return False
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        status: str = "closed"
    ) -> bool:
        """
        Mark a position as closed.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            realized_pnl: Realized P&L
            status: "closed" or "resolved"
            
        Returns:
            True if updated successfully
        """
        if self.db is None:
            return False
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                await session.execute(
                    text("""
                        UPDATE paper_positions
                        SET closed_at = NOW(),
                            exit_price = :exit_price,
                            realized_pnl = :realized_pnl,
                            status = :status
                        WHERE id = :id
                    """),
                    {
                        "id": position_id,
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl,
                        "status": status
                    }
                )
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    async def load_open_positions(self) -> List[PaperPosition]:
        """
        Load all open positions from database.
        
        Returns:
            List of open positions
        """
        if self.db is None:
            return []
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                result = await session.execute(
                    text("""
                        SELECT id, market_id, market_name, side, size, entry_price, opened_at, signal_id, signal_score
                        FROM paper_positions
                        WHERE status = 'open'
                    """)
                )
                rows = result.fetchall()
                
                positions = []
                for row in rows:
                    positions.append(PaperPosition(
                        id=row[0],
                        market_id=row[1],
                        market_name=row[2] or row[1],
                        side=row[3],
                        size=row[4],
                        entry_price=row[5],
                        current_price=row[5],  # Use entry price as current
                        opened_at=row[6],
                        signal_id=row[7],
                        signal_score=row[8]
                    ))
                
                return positions
                
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return []
    
    async def save_trade(self, trade: PaperTrade) -> bool:
        """
        Save an individual trade to database.
        
        Args:
            trade: Trade to save
            
        Returns:
            True if saved successfully
        """
        if self.db is None:
            return False
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                holding_seconds = None
                if trade.holding_period:
                    holding_seconds = trade.holding_period.total_seconds()
                
                await session.execute(
                    text("""
                        INSERT INTO paper_trades 
                        (id, position_id, market_id, market_name, side, action, size, price, fees, total, 
                         signal_id, signal_score, realized_pnl, holding_period_seconds, timestamp)
                        VALUES (:id, :position_id, :market_id, :market_name, :side, :action, :size, :price, 
                                :fees, :total, :signal_id, :signal_score, :realized_pnl, :holding_seconds, :timestamp)
                    """),
                    {
                        "id": trade.id,
                        "position_id": trade.position_id,
                        "market_id": trade.market_id,
                        "market_name": trade.market_name,
                        "side": trade.side,
                        "action": trade.action,
                        "size": trade.size,
                        "price": trade.price,
                        "fees": trade.fees,
                        "total": trade.total,
                        "signal_id": trade.signal_id,
                        "signal_score": trade.signal_score,
                        "realized_pnl": trade.realized_pnl,
                        "holding_seconds": holding_seconds,
                        "timestamp": trade.timestamp
                    }
                )
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return False
    
    async def load_trades(self, limit: int = 100) -> List[PaperTrade]:
        """
        Load trade history from database.
        
        Args:
            limit: Maximum trades to load
            
        Returns:
            List of trades
        """
        if self.db is None:
            return []
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                from datetime import timedelta
                
                result = await session.execute(
                    text("""
                        SELECT id, position_id, market_id, market_name, side, action, size, price, 
                               fees, total, signal_id, signal_score, realized_pnl, holding_period_seconds, timestamp
                        FROM paper_trades
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                )
                rows = result.fetchall()
                
                trades = []
                for row in rows:
                    holding_period = None
                    if row[13]:
                        holding_period = timedelta(seconds=row[13])
                    
                    trades.append(PaperTrade(
                        id=row[0],
                        position_id=row[1],
                        market_id=row[2],
                        market_name=row[3] or row[2],
                        side=row[4],
                        action=row[5],
                        size=row[6],
                        price=row[7],
                        fees=row[8],
                        total=row[9],
                        signal_id=row[10],
                        signal_score=row[11],
                        realized_pnl=row[12],
                        holding_period=holding_period,
                        timestamp=row[14]
                    ))
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")
            return []
    
    async def save_snapshot(self, state: PortfolioState) -> bool:
        """
        Save a portfolio snapshot for equity curve tracking.
        
        Args:
            state: Portfolio state to snapshot
            
        Returns:
            True if saved successfully
        """
        if self.db is None:
            return False
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                await session.execute(
                    text("""
                        INSERT INTO paper_snapshots 
                        (timestamp, cash_balance, positions_value, total_value, unrealized_pnl, realized_pnl)
                        VALUES (:timestamp, :cash_balance, :positions_value, :total_value, :unrealized_pnl, :realized_pnl)
                    """),
                    {
                        "timestamp": state.timestamp,
                        "cash_balance": state.cash_balance,
                        "positions_value": state.positions_value,
                        "total_value": state.total_value,
                        "unrealized_pnl": state.unrealized_pnl,
                        "realized_pnl": state.realized_pnl
                    }
                )
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
    
    async def load_snapshots(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PortfolioSnapshot]:
        """
        Load portfolio snapshots from database.
        
        Args:
            start: Optional start timestamp
            end: Optional end timestamp
            limit: Maximum snapshots to load
            
        Returns:
            List of portfolio snapshots
        """
        if self.db is None:
            return []
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                query = "SELECT id, timestamp, cash_balance, positions_value, total_value, unrealized_pnl, realized_pnl FROM paper_snapshots"
                params = {"limit": limit}
                
                conditions = []
                if start:
                    conditions.append("timestamp >= :start")
                    params["start"] = start
                if end:
                    conditions.append("timestamp <= :end")
                    params["end"] = end
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp ASC LIMIT :limit"
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                return [
                    PortfolioSnapshot(
                        id=row[0],
                        timestamp=row[1],
                        cash_balance=row[2],
                        positions_value=row[3],
                        total_value=row[4],
                        unrealized_pnl=row[5],
                        realized_pnl=row[6]
                    )
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to load snapshots: {e}")
            return []
    
    async def reset(self, initial_balance: float) -> bool:
        """
        Reset paper trading state.
        
        Clears all positions, trades, and snapshots.
        
        Args:
            initial_balance: New initial balance
            
        Returns:
            True if reset successfully
        """
        if self.db is None:
            return False
        
        try:
            async with self.db._session_factory() as session:
                from sqlalchemy import text
                
                # Delete all paper trading data
                await session.execute(text("DELETE FROM paper_snapshots"))
                await session.execute(text("DELETE FROM paper_trades"))
                await session.execute(text("DELETE FROM paper_positions"))
                await session.execute(text("DELETE FROM paper_portfolio"))
                
                # Insert fresh portfolio
                await session.execute(
                    text("""
                        INSERT INTO paper_portfolio (balance, initial_balance, realized_pnl, total_fees)
                        VALUES (:balance, :initial_balance, 0, 0)
                    """),
                    {"balance": initial_balance, "initial_balance": initial_balance}
                )
                
                await session.commit()
                logger.info(f"Reset paper trading with balance ${initial_balance:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to reset paper trading: {e}")
            return False

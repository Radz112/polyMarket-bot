"""
Position history tracking and analytics.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import statistics

from src.database.postgres import DatabaseManager
from src.execution.positions.types import (
    Position,
    ClosedPosition,
    PositionSnapshot,
    PositionAnalytics,
)

logger = logging.getLogger(__name__)


class PositionHistory:
    """
    Tracks position history for analytics and auditing.
    
    Provides:
    - Position state snapshots over time
    - Closed position history
    - Entry/exit analytics
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        Initialize history tracker.
        
        Args:
            db: Database manager for persistence
        """
        self.db = db
        
        # In-memory storage
        self._snapshots: dict[str, List[PositionSnapshot]] = {}  # position_id -> snapshots
        self._closed_positions: List[ClosedPosition] = []
    
    def record_snapshot(self, position: Position) -> None:
        """
        Record current position state for history.
        
        Args:
            position: Position to snapshot
        """
        snapshot = PositionSnapshot(
            id=None,
            position_id=position.id,
            timestamp=datetime.utcnow(),
            price=position.current_price,
            bid=position.current_bid,
            ask=position.current_ask,
            unrealized_pnl=position.unrealized_pnl,
            unrealized_pnl_pct=position.unrealized_pnl_pct
        )
        
        if position.id not in self._snapshots:
            self._snapshots[position.id] = []
        
        self._snapshots[position.id].append(snapshot)
    
    def record_snapshots(self, positions: List[Position]) -> None:
        """Record snapshots for multiple positions."""
        for position in positions:
            self.record_snapshot(position)
    
    def record_closed_position(self, closed: ClosedPosition) -> None:
        """
        Record a closed position.
        
        Args:
            closed: Closed position record
        """
        self._closed_positions.append(closed)
    
    def get_position_history(
        self,
        position_id: str
    ) -> List[PositionSnapshot]:
        """
        Get price/P&L history for a position.
        
        Args:
            position_id: Position to get history for
            
        Returns:
            List of snapshots ordered by time
        """
        snapshots = self._snapshots.get(position_id, [])
        return sorted(snapshots, key=lambda s: s.timestamp)
    
    def get_closed_positions(
        self,
        start: datetime = None,
        end: datetime = None,
        category: str = None
    ) -> List[ClosedPosition]:
        """
        Get historical closed positions.
        
        Args:
            start: Optional start timestamp
            end: Optional end timestamp
            category: Optional category filter
            
        Returns:
            List of closed positions
        """
        results = self._closed_positions
        
        if start:
            results = [c for c in results if c.exit_time >= start]
        
        if end:
            results = [c for c in results if c.exit_time <= end]
        
        if category:
            results = [c for c in results if c.position.category == category]
        
        return sorted(results, key=lambda c: c.exit_time, reverse=True)
    
    def get_position_analytics(
        self,
        position_id: str
    ) -> Optional[PositionAnalytics]:
        """
        Generate analytics for a position.
        
        Args:
            position_id: Position to analyze
            
        Returns:
            PositionAnalytics or None if insufficient data
        """
        snapshots = self.get_position_history(position_id)
        
        if len(snapshots) < 2:
            return None
        
        prices = [s.price for s in snapshots]
        pnls = [s.unrealized_pnl for s in snapshots]
        
        # Find max gain/loss
        max_gain = max(pnls)
        max_loss = min(pnls)
        
        max_gain_idx = pnls.index(max_gain)
        max_loss_idx = pnls.index(max_loss)
        
        max_gain_time = snapshots[max_gain_idx].timestamp if max_gain > 0 else None
        max_loss_time = snapshots[max_loss_idx].timestamp if max_loss < 0 else None
        
        # Calculate price volatility
        if len(prices) >= 2:
            price_volatility = statistics.stdev(prices)
        else:
            price_volatility = 0
        
        # Entry timing score
        # Compare entry price to subsequent prices
        first_price = prices[0]
        avg_price = statistics.mean(prices)
        best_price_for_long = min(prices)
        
        # For YES position (buying), lower entry is better
        # Score: how close was entry to the best price?
        if first_price == best_price_for_long:
            entry_score = 100
        elif avg_price != best_price_for_long:
            entry_score = max(0, 100 - (
                (first_price - best_price_for_long) / 
                (avg_price - best_price_for_long + 0.001) * 100
            ))
        else:
            entry_score = 50
        
        # Current vs optimal exit
        current_pnl = pnls[-1] if pnls else 0
        optimal_pnl = max_gain
        current_vs_optimal = current_pnl - optimal_pnl
        
        return PositionAnalytics(
            position_id=position_id,
            entry_timing_score=min(100, max(0, entry_score)),
            max_gain=max_gain,
            max_gain_time=max_gain_time,
            max_loss=max_loss,
            max_loss_time=max_loss_time,
            current_vs_optimal=current_vs_optimal,
            price_volatility=price_volatility,
            snapshots_count=len(snapshots)
        )
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics across all history.
        
        Returns:
            Dict with summary stats
        """
        closed = self._closed_positions
        
        if not closed:
            return {
                "total_closed": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_holding_period_hours": 0
            }
        
        total_pnl = sum(c.realized_pnl for c in closed)
        winners = [c for c in closed if c.realized_pnl > 0]
        win_rate = len(winners) / len(closed) if closed else 0
        
        holding_periods = [c.holding_period for c in closed]
        avg_holding = sum(holding_periods, timedelta()) / len(holding_periods)
        
        return {
            "total_closed": len(closed),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_holding_period_hours": avg_holding.total_seconds() / 3600
        }
    
    def clear(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._closed_positions.clear()


async def save_position_snapshot_to_db(
    db: DatabaseManager,
    snapshot: PositionSnapshot
) -> bool:
    """
    Save a position snapshot to database.
    
    Args:
        db: Database manager
        snapshot: Snapshot to save
        
    Returns:
        True if saved successfully
    """
    try:
        from sqlalchemy import text
        
        async with db._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO position_snapshots 
                    (position_id, timestamp, price, bid, ask, unrealized_pnl, unrealized_pnl_pct)
                    VALUES (:position_id, :timestamp, :price, :bid, :ask, :unrealized_pnl, :unrealized_pnl_pct)
                """),
                {
                    "position_id": snapshot.position_id,
                    "timestamp": snapshot.timestamp,
                    "price": snapshot.price,
                    "bid": snapshot.bid,
                    "ask": snapshot.ask,
                    "unrealized_pnl": snapshot.unrealized_pnl,
                    "unrealized_pnl_pct": snapshot.unrealized_pnl_pct
                }
            )
            await session.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to save position snapshot: {e}")
        return False

"""
Market resolution handler for paper trading.

Monitors markets for resolution and settles positions accordingly.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict

from src.api.clob_client import ClobClient
from src.execution.paper.types import PaperPosition, PaperTrade

logger = logging.getLogger(__name__)


class ResolutionHandler:
    """
    Handles market resolutions for paper trading positions.
    
    When a market resolves:
    - YES resolves to $1.00
    - NO resolves to $0.00
    
    Positions are settled at these final prices.
    """
    
    def __init__(
        self,
        api_client: Optional[ClobClient] = None
    ):
        """
        Initialize the resolution handler.
        
        Args:
            api_client: Polymarket API client for checking resolutions
        """
        self.api_client = api_client
        
        # Cache of known resolutions
        self._resolutions: Dict[str, str] = {}  # market_id -> outcome
    
    async def check_resolutions(
        self,
        positions: Dict[str, PaperPosition]
    ) -> List[tuple]:
        """
        Check if any markets with open positions have resolved.
        
        Args:
            positions: Dict of position_id -> PaperPosition
            
        Returns:
            List of (market_id, outcome) tuples for resolved markets
        """
        resolved = []
        
        # Get unique market IDs from positions
        market_ids = set(pos.market_id for pos in positions.values())
        
        for market_id in market_ids:
            # Check cache first
            if market_id in self._resolutions:
                resolved.append((market_id, self._resolutions[market_id]))
                continue
            
            # Check API if available
            if self.api_client:
                try:
                    outcome = await self._check_market_resolution(market_id)
                    if outcome:
                        self._resolutions[market_id] = outcome
                        resolved.append((market_id, outcome))
                except Exception as e:
                    logger.warning(f"Error checking resolution for {market_id}: {e}")
        
        return resolved
    
    async def _check_market_resolution(self, market_id: str) -> Optional[str]:
        """
        Check if a market has resolved via API.
        
        Args:
            market_id: Market to check
            
        Returns:
            "YES" or "NO" if resolved, None if still open
        """
        try:
            market_data = await self.api_client.get_market(market_id)
            
            # Check if market is resolved
            if market_data.get("resolved", False):
                outcome = market_data.get("outcome")
                if outcome in ["YES", "NO"]:
                    logger.info(f"Market {market_id} resolved to {outcome}")
                    return outcome
            
            return None
        except Exception as e:
            logger.warning(f"Failed to check market {market_id}: {e}")
            return None
    
    def get_settlement_price(
        self,
        position_side: str,
        resolution_outcome: str
    ) -> float:
        """
        Get the settlement price for a position given resolution outcome.
        
        Args:
            position_side: "YES" or "NO" (what the position holds)
            resolution_outcome: "YES" or "NO" (what the market resolved to)
            
        Returns:
            1.0 if position wins, 0.0 if position loses
        """
        if position_side == resolution_outcome:
            return 1.0  # Position wins
        else:
            return 0.0  # Position loses
    
    async def handle_resolution(
        self,
        position: PaperPosition,
        outcome: str
    ) -> tuple:
        """
        Handle a single market resolution for a position.
        
        Args:
            position: The position to settle
            outcome: "YES" or "NO" resolution
            
        Returns:
            Tuple of (settlement_price, realized_pnl)
        """
        settlement_price = self.get_settlement_price(position.side, outcome)
        
        # Calculate P&L
        proceeds = position.size * settlement_price
        cost_basis = position.cost_basis
        realized_pnl = proceeds - cost_basis
        
        logger.info(
            f"Resolved position {position.id}: {position.side} @ {outcome}, "
            f"P&L: ${realized_pnl:.4f}"
        )
        
        return settlement_price, realized_pnl
    
    def set_resolution(self, market_id: str, outcome: str) -> None:
        """
        Manually set a market resolution (for testing or manual override).
        
        Args:
            market_id: Market ID
            outcome: "YES" or "NO"
        """
        if outcome not in ["YES", "NO"]:
            raise ValueError(f"Invalid outcome: {outcome}")
        
        self._resolutions[market_id] = outcome
        logger.info(f"Set resolution for {market_id} to {outcome}")
    
    def clear_resolution(self, market_id: str) -> None:
        """Remove a cached resolution."""
        if market_id in self._resolutions:
            del self._resolutions[market_id]
    
    def get_cached_resolutions(self) -> Dict[str, str]:
        """Get all cached resolutions."""
        return self._resolutions.copy()


async def monitor_pending_resolutions(
    positions: Dict[str, PaperPosition],
    api_client: ClobClient,
    check_interval_minutes: int = 5
) -> None:
    """
    Background task to monitor markets near resolution.
    
    This is a coroutine that should be run as a background task.
    
    Args:
        positions: Dict of active positions to monitor
        api_client: API client for checking resolutions
        check_interval_minutes: How often to check (in minutes)
    """
    import asyncio
    
    handler = ResolutionHandler(api_client)
    
    while True:
        try:
            resolved = await handler.check_resolutions(positions)
            
            if resolved:
                logger.info(f"Found {len(resolved)} resolved markets")
                # Note: Actual settlement should be handled by the caller
                # This just detects resolutions
            
            await asyncio.sleep(check_interval_minutes * 60)
            
        except asyncio.CancelledError:
            logger.info("Resolution monitor cancelled")
            break
        except Exception as e:
            logger.error(f"Error in resolution monitor: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

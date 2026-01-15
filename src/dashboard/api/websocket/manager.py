"""
WebSocket manager for real-time updates.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Channels:
    - signals: New/updated signals
    - positions: Position updates
    - portfolio: Portfolio value updates
    - risk: Risk alerts
    """
    
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = defaultdict(list)
    
    async def connect(self, channel: str, websocket: WebSocket) -> None:
        """
        Accept WebSocket connection on channel.
        
        Args:
            channel: Channel name (signals, positions, portfolio, risk)
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.connections[channel].append(websocket)
        logger.info(f"WebSocket connected to {channel}")
    
    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        """
        Remove WebSocket from channel.
        
        Args:
            channel: Channel name
            websocket: WebSocket to remove
        """
        if websocket in self.connections[channel]:
            self.connections[channel].remove(websocket)
            logger.info(f"WebSocket disconnected from {channel}")
    
    async def broadcast(self, channel: str, message: Dict[str, Any]) -> None:
        """
        Broadcast message to all connections on channel.
        
        Args:
            channel: Channel name
            message: JSON-serializable message
        """
        dead_connections = []
        
        for connection in self.connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(channel, conn)
    
    async def send_personal(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific connection.
        
        Args:
            websocket: Target WebSocket
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            return False
    
    def get_connection_count(self, channel: str = None) -> int:
        """
        Get number of active connections.
        
        Args:
            channel: Specific channel or None for all
            
        Returns:
            Connection count
        """
        if channel:
            return len(self.connections[channel])
        return sum(len(conns) for conns in self.connections.values())
    
    async def broadcast_signal(self, signal_type: str, signal: Dict[str, Any]) -> None:
        """Broadcast signal event."""
        await self.broadcast("signals", {
            "type": signal_type,
            "signal": signal
        })
    
    async def broadcast_position(
        self,
        event_type: str,
        position: Dict[str, Any],
        pnl: float = None
    ) -> None:
        """Broadcast position event."""
        message = {
            "type": event_type,
            "position": position
        }
        if pnl is not None:
            message["pnl"] = pnl
        await self.broadcast("positions", message)
    
    async def broadcast_portfolio(
        self,
        value: float,
        pnl: float,
        pnl_pct: float
    ) -> None:
        """Broadcast portfolio update."""
        await self.broadcast("portfolio", {
            "type": "portfolio_update",
            "value": value,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    async def broadcast_risk_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast risk alert."""
        await self.broadcast("risk", {
            "type": "risk_alert",
            "alert": alert
        })
    
    async def broadcast_breaker_tripped(self, breaker: str, reason: str) -> None:
        """Broadcast circuit breaker trip."""
        await self.broadcast("risk", {
            "type": "breaker_tripped",
            "breaker": breaker,
            "reason": reason
        })


# Global WebSocket manager instance
ws_manager = WebSocketManager()

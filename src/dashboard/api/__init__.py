"""
Dashboard API module.
"""
from .main import app
from .websocket import ws_manager, WebSocketManager
from .dependencies import app_state, get_config, get_position_manager, get_order_manager

__all__ = [
    "app",
    "ws_manager",
    "WebSocketManager",
    "app_state",
    "get_config",
    "get_position_manager",
    "get_order_manager",
]

"""
Dependency injection for API routes.
"""
from functools import lru_cache
from typing import Optional

from src.config.settings import Config
from src.execution.positions import PositionManager
from src.execution.orders import OrderManager
from src.database.postgres import DatabaseManager


class AppState:
    """
    Application state container.
    
    Holds references to managers and services.
    """
    
    def __init__(self):
        self.config: Optional[Config] = None
        self.db: Optional[DatabaseManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.start_time: float = 0
        self.trading_halted: bool = False
        self.halt_reasons: list = []
    
    def initialize(self, config: Config):
        """Initialize with configuration."""
        import time
        self.config = config
        self.start_time = time.time()

        # Initialize database
        self.db = DatabaseManager(config)

        # Initialize clients
        self.live_client = None
        if not config.paper_trading:
            from src.api.live_client import LiveTradingClient
            is_valid, msg = config.validate_live_trading()
            if is_valid:
                self.live_client = LiveTradingClient(
                    private_key=config.polymarket_private_key,
                    funder=config.polymarket_funder,
                    signature_type=config.polymarket_signature_type,
                )
            else:
                print(f"⚠️ Live trading config invalid for dashboard: {msg}")

        # Initialize managers
        self.position_manager = PositionManager()
        self.order_manager = OrderManager(
            config=config,
            position_manager=self.position_manager,
            is_paper=config.paper_trading,
            api_client=self.live_client
        )


# Global app state
app_state = AppState()


def get_config() -> Config:
    """Get application configuration."""
    if app_state.config is None:
        app_state.config = Config()
    return app_state.config


def get_db() -> DatabaseManager:
    """Get database manager."""
    if app_state.db is None:
        config = get_config()
        app_state.db = DatabaseManager(config)
    return app_state.db


def get_position_manager() -> PositionManager:
    """Get position manager."""
    if app_state.position_manager is None:
        app_state.position_manager = PositionManager()
    return app_state.position_manager


def get_order_manager() -> OrderManager:
    """Get order manager."""
    if app_state.order_manager is None:
        config = get_config()
        # Ensure app_state is initialized if it wasn't
        if app_state.config is None:
            app_state.initialize(config)
    return app_state.order_manager

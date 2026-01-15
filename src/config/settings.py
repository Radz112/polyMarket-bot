from typing import List, Optional, Tuple, Type, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
import yaml
import os

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source that loads variables from a YAML file
    at the project's config/config.yaml location.
    """
    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        pass

    def __call__(self) -> Dict[str, Any]:
        config_file = os.getenv("CONFIG_FILE", "config/config.yaml")
        if os.path.exists(config_file):
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        return {}

class Config(BaseSettings):
    # Environment
    env: str = Field("development", description="Environment: development, staging, production")
    debug: bool = False

    # Polymarket API
    polymarket_api_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polymarket_private_key: Optional[str] = None
    polymarket_funder: Optional[str] = None
    polymarket_signature_type: int = 1  # 0=EOA, 1=PolyProxy/Gnosis, 2=Gnosis

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "polymarket_bot"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_pool_size: int = 10
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Trading parameters
    min_divergence_threshold: float = 0.02
    min_signal_score: float = 60
    min_correlation_confidence: float = 0.8
    max_position_size_pct: float = 0.05
    max_daily_loss_pct: float = 0.10
    paper_trading: bool = True
    initial_capital: float = 10000
    
    # Alerting
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/bot.log"
    
    def validate_live_trading(self) -> tuple[bool, str]:
        """
        Validate configuration for live trading.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.paper_trading:
            return True, "Paper trading mode - no validation needed"
        
        if not self.polymarket_private_key:
            return False, "POLYMARKET_PRIVATE_KEY is required for live trading"
        
        if not self.polymarket_private_key.startswith("0x"):
            return False, "POLYMARKET_PRIVATE_KEY must start with 0x"
        
        if len(self.polymarket_private_key) != 66:  # 0x + 64 hex chars
            return False, "POLYMARKET_PRIVATE_KEY must be 66 characters (0x + 64 hex)"
        
        return True, "Configuration valid for live trading"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

# Singleton instance
config = Config()

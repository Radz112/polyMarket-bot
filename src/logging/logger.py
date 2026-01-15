import logging
import sys
from datetime import datetime
from typing import Any, Dict
import json
from src.config.settings import Config

class JSONFormatter(logging.Formatter):
    """JSON log formatter for production"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)

class PrettyFormatter(logging.Formatter):
    """Pretty formatter for development"""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Simple message
        msg = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} {record.getMessage()}"
        
        # Add extra data if present (indented)
        if hasattr(record, "extra_data"):
            extra = record.extra_data
            msg += f"\n  {self.COLORS['DEBUG']}{json.dumps(extra, indent=2)}{self.RESET}"
            
        return msg

class BotLogger:
    """Custom logger with context support"""
    
    def __init__(self, name: str = "polymarket_bot"):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False # Prevent double logging if root logger is configured
        self._context: Dict[str, Any] = {}
        self._setup_done = False
    
    def setup(self, config: Config):
        """Setup logging based on config"""
        if self._setup_done:
            return

        self.logger.setLevel(config.log_level)
        self.logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        if config.env == "production":
            console.setFormatter(JSONFormatter())
        else:
            console.setFormatter(PrettyFormatter())
        self.logger.addHandler(console)
        
        # File handler
        if config.log_file:
            from logging.handlers import RotatingFileHandler
            try:
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    config.log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
                file_handler.setFormatter(JSONFormatter())
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to setup file logging: {e}")
        
        self._setup_done = True
    
    def with_context(self, **kwargs) -> "BotLogger":
        """Return logger with additional context"""
        new_logger = BotLogger(self.logger.name)
        new_logger.logger = self.logger
        new_logger._context = {**self._context, **kwargs}
        new_logger._setup_done = self._setup_done
        return new_logger
    
    def _log(self, level: int, msg: str, **kwargs):
        # Merge context
        extra_data = {**self._context, **kwargs}
        if extra_data:
            extra = {"extra_data": extra_data}
        else:
            extra = {}
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def trade(self, action: str, **kwargs):
        """Log trade (always logged)"""
        self.info(f"TRADE: {action}", trade=True, **kwargs)
    
    def signal(self, signal_type: str, **kwargs):
        """Log signal"""
        self.info(f"SIGNAL: {signal_type}", signal=True, **kwargs)

# Singleton
logger = BotLogger()

def setup_logging(config: Config):
    """Initialize logging"""
    logger.setup(config)

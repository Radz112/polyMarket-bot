import logging
import logging.handlers
import os

from .formatters import JSONFormatter, PrettyFormatter
from src.config import config

def setup_handlers(logger: logging.Logger):
    # Ensure log directory
    log_file = config.log_file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File Handler (JSON)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=7
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.DEBUG) # Catch all in file
        logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    if config.env == "development":
        console_handler.setFormatter(PrettyFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    else:
        console_handler.setFormatter(JSONFormatter())
    
    console_handler.setLevel(config.log_level)
    logger.addHandler(console_handler)

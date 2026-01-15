import logging
import sys
from pathlib import Path

def setup_logging(level: int = logging.INFO):
    """Sets up standard logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_dir / "bot.log")
    file_handler.setFormatter(formatter)

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set external libraries to WARNING to reduce noise
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

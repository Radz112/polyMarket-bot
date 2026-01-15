import logging
import json
import coloredlogs
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Formats logs as JSON."""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        
        # Add extra fields (context)
        if hasattr(record, "context"):
            log_obj.update(record.context)
            
        # Add special fields if trade or signal
        if hasattr(record, "trade_data"):
            log_obj["trade"] = record.trade_data
        if hasattr(record, "signal_data"):
            log_obj["signal"] = record.signal_data
            
        # Add exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

class PrettyFormatter(logging.Formatter):
    """Formats logs for console with colors via coloredlogs (if configured)."""
    # Note: simple wrapper, actual coloring handled by coloredlogs install
    def format(self, record):
        msg = super().format(record)
        context = getattr(record, "context", {})
        if context:
            msg += f" | {context}"
        return msg
